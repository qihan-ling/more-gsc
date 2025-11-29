"""
Diagnostic to check why SAP activates wrong bindings

Compares how the same sentence "N Vi P N" is processed
in speedup vs speedup_sap versions
"""

import numpy as np

# Test sentence: "N Vi P N"
test_words = ['N', 'Vi', 'P', 'N']

print("="*70)
print("DIAGNOSING WRONG ACTIVATIONS IN SAP VERSION")
print("="*70)

for module_name in ['only_gscnet_speedup', 'only_gscnet_speedup_sap']:
    print(f"\n{'='*70}")
    print(f"Testing: {module_name}")
    print(f"{'='*70}")

    if module_name == 'only_gscnet_speedup':
        import only_gscnet_speedup as gsc
        model_file = 'g1_ds_speedup_model_copy_trainjax.pkl'
    else:
        import only_gscnet_speedup_sap as gsc
        model_file = 'ds_jax_sap_test_on_g1_model.pkl'

    # Load model
    try:
        net = gsc.load_model(model_file)
    except:
        print(f"  ❌ Model file {model_file} not found, skipping")
        continue

    print(f"\nModel info:")
    print(f"  max_sent_len: {net.hg.opts['max_sent_len']}")
    print(f"  num_bindings: {net.num_bindings}")
    print(f"  num_roles: {net.num_roles}")

    # Reset network
    net.reset(mu=net.ep, sd=0.01)

    print(f"\nProcessing sentence: {' '.join(test_words)}")

    # Process word by word and check external inputs
    for wi, word in enumerate(test_words):
        print(f"\n  Word {wi+1}: '{word}' at position {wi+1}")

        # Run the word
        net.run_word(word, wi + 1, log_trace=False)

        # Check what external input was set
        extC_active = np.where(net.extC > 0)[0]
        print(f"    Active external inputs ({len(extC_active)} bindings):")

        if len(extC_active) > 0:
            # Show first 5 active bindings
            for idx in extC_active[:5]:
                bname = net.binding_names[idx]
                ext_val = net.extC[idx]
                print(f"      [{idx}] {bname}: {ext_val:.4f}")

            if len(extC_active) > 5:
                print(f"      ... and {len(extC_active) - 5} more")
        else:
            print(f"      ❌ NO EXTERNAL INPUTS SET! This is the problem!")

    # Run wrapup
    print(f"\n  Running wrapup...")
    net.run_wrapup(log_trace=False)

    # Check final activations at key roles
    print(f"\n  Final activations at role (3,2):")

    # Find bindings at role (3,2)
    role_idx = net.hg.roles.role_name_to_idx.get('(3,2)', None)
    if role_idx is not None:
        bindings_at_role = net.role_to_binding_indices[role_idx]

        # Get activations
        actC_at_role = net.actC[bindings_at_role]

        # Sort by activation
        sorted_idx = np.argsort(-actC_at_role)

        print(f"    Top 5 activated bindings at (3,2):")
        for i in sorted_idx[:5]:
            bname = net.binding_names[bindings_at_role[i]]
            act = actC_at_role[i]
            print(f"      {bname}: {act:.4f}")

    print()

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)
print("""
If SAP version shows:
  - NO EXTERNAL INPUTS or wrong external inputs
    → Problem in run_word() or set_discrete_state()

  - External inputs correct but wrong final activations
    → Problem in dynamics (run_wrapup, update_stateC, HGradC)

  - Wrong bindings at roles (empty markers like @ or # instead of words)
    → Problem with max_sent_len mismatch or binding index mapping
""")
print("="*70)
