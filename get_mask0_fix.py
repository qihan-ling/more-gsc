# BATCH-BASED FIX for get_mask0()
# Replace lines 3252-3313 with this code

# FIXED: Build mask0 in BATCHES to avoid holding 10+ billion entries in memory
# Process roles in batches, building and accumulating CSR matrices incrementally
mask0 = sparse.csr_matrix(self.WC.shape, dtype=np.float64)

batch_size = 10  # Process 10 roles at a time
non_terminal_roles = [ri for ri in range(len(self.hg.role_names))
                     if not self.hg.roles.role_is_terminal[ri]]
total_roles = len(non_terminal_roles)

for batch_start in range(0, total_roles, batch_size):
    batch_end = min(batch_start + batch_size, total_roles)
    role_indices = non_terminal_roles[batch_start:batch_end]

    if batch_start % 50 == 0:
        print(f"      Processing batch {batch_start//batch_size + 1}/{(total_roles + batch_size - 1)//batch_size} (roles {batch_start}-{batch_end}/{total_roles})...")

    # Collect pairs for this batch only
    row_list = []
    col_list = []

    for ri in role_indices:
        indices = self.get_role_and_daughter_indices_fast(ri)
        if indices != None:
            idx = np.array(indices['self'])
            idx_l = np.array(indices['l'])
            idx_r = np.array(indices['r'])

            # idx × idx (self-role)
            rows_self, cols_self = np.meshgrid(idx, idx, indexing='ij')
            row_list.append(rows_self.ravel())
            col_list.append(cols_self.ravel())

            # idx × idx_l (parent-left) + symmetric
            rows_pl, cols_pl = np.meshgrid(idx, idx_l, indexing='ij')
            row_list.extend([rows_pl.ravel(), cols_pl.ravel()])
            col_list.extend([cols_pl.ravel(), rows_pl.ravel()])

            # idx × idx_r (parent-right) + symmetric
            rows_pr, cols_pr = np.meshgrid(idx, idx_r, indexing='ij')
            row_list.extend([rows_pr.ravel(), cols_pr.ravel()])
            col_list.extend([cols_pr.ravel(), rows_pr.ravel()])

            # Sister harmony (if enabled)
            if self.train_opts['update_sister_harmony']:
                rows_s, cols_s = np.meshgrid(idx_l, idx_r, indexing='ij')
                row_list.extend([rows_s.ravel(), cols_s.ravel()])
                col_list.extend([cols_s.ravel(), rows_s.ravel()])

    # Build COO for this batch and add to mask0
    if row_list:
        batch_rows = np.concatenate(row_list)
        batch_cols = np.concatenate(col_list)
        batch_data = np.ones(len(batch_rows), dtype=np.float64)

        batch_coo = sparse.coo_matrix(
            (batch_data, (batch_rows, batch_cols)),
            shape=self.WC.shape,
            dtype=np.float64
        )

        # Add to cumulative mask (CSR addition handles duplicates)
        mask0 = mask0 + batch_coo.tocsr()

        # Explicitly free batch memory
        del batch_rows, batch_cols, batch_data, batch_coo, row_list, col_list

# Normalize: set all non-zero values to 1 (removing duplicate counts)
mask0.data = np.ones_like(mask0.data)

print(f"      Total mask0 construction: {time.time() - t_start:.2f}s")
print(f"      mask0 has {mask0.nnz:,} non-zero entries")
