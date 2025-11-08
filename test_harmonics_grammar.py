import only_datastructure_speedup as ds
with open('filtered_sm5.grammar', 'r') as f:
    PCFG_sap = f.read()
hg = ds.HarmonicGrammar(pcfg=PCFG_sap, root='S', max_sent_len=24)

print(hg.roles.role_tuples.shape)        # Should print: (300, 2)
# Should print: number of terminal roles
print(hg.roles.role_is_terminal.sum())
# Should print: number of terminal fillers
print(hg.g.filler_is_terminal.sum())

# Use in training
for ri in range(len(hg.role_names)):
    lv, pos = hg.roles.role_tuples[ri]  # Fast!
    if hg.roles.role_is_terminal[ri]:   # Fast!
        print(f"Terminal role: ({lv}, {pos})")
