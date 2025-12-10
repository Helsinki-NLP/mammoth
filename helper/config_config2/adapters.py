from .utils import load_yaml
from .schema import (register_schema, register_command_io, register_command_template_extras)

# PURPOSE: Expand adapter ID spaces (LANGUAGE/GROUP/FULL) into concrete IDs and attach per-task adapter selections.
def adapter_config(opts):
     start = time.time()

     cc_opts = opts.in_config[0]['config_config']
     cc = cc_opts
     
     # Require adapters section
     adapters = cfg.get("adapters")
     if not isinstance(adapters, dict):
          raise UserConfigError("Missing 'adapters' section in YAML. Add adapters: {encoder: {...}, decoder: {...}}.")
     
     # Require groups (used for GROUP id-space)
     if "groups" not in cc or not isinstance(cc["groups"], dict):
          raise UserConfigError("Missing 'config_config.groups' mapping (needed for GROUP adapter ids).")
     
     # If there is no 'adapters' section, nothing to do.
     if 'adapters' not in opts.in_config[0]:
          logger.warning('No adapter configuration, skipping this step')
          return

     # Collect language and group sets.
     src_langs, tgt_langs = _get_langs(opts)
     src_groups = list(sorted(set(cc_opts['groups'][src] for src in src_langs)))
     tgt_groups = list(sorted(set(cc_opts['groups'][tgt] for tgt in tgt_langs)))

     # Adapter specs are keyed by name under 'encoder' and 'decoder'.
     encoder_adapters = opts.in_config[0]['adapters'].get('encoder', [])
     decoder_adapters = opts.in_config[0]['adapters'].get('decoder', [])

     # Ensure each task has its own adapters list present.
     for task_key, task_config in opts.in_config[0]['tasks'].items():
         if 'adapters' not in task_config:
             task_config['adapters'] = {'encoder': [], 'decoder': []}
             
     # Expand encoder adapter id spaces to concrete ids and attach to tasks.
     if len(encoder_adapters) > 0:
         for adapter_name, adapter_config in sorted(encoder_adapters.items()):
             if adapter_config['ids'] == 'LANGUAGE':
                 adapter_config['ids'] = list(src_langs)
                 for task_key, task_config in opts.in_config[0]['tasks'].items():
                     task_src, task_tgt = task_config['src_tgt'].split('-')
                     task_config['adapters']['encoder'].append([adapter_name, task_src])
             elif adapter_config['ids'] == 'GROUP':
                 adapter_config['ids'] = list(src_groups)
                 for task_key, task_config in opts.in_config[0]['tasks'].items():
                     task_src, task_tgt = task_config['src_tgt'].split('-')
                     task_config['adapters']['encoder'].append([adapter_name, cc_opts['groups'][task_src]])
             elif adapter_config['ids'] == 'FULL':
                 adapter_config['ids'] = ['full']
                 for task_key, task_config in opts.in_config[0]['tasks'].items():
                     task_config['adapters']['encoder'].append([adapter_name, 'full'])
                     
     # Expand decoder adapter id spaces and attach.
     if len(decoder_adapters) > 0:
         for adapter_name, adapter_config in sorted(decoder_adapters.items()):
             if adapter_config['ids'] == 'LANGUAGE':
                 adapter_config['ids'] = list(tgt_langs)
                 for task_key, task_config in opts.in_config[0]['tasks'].items():
                     task_src, task_tgt = task_config['src_tgt'].split('-')
                     task_config['adapters']['decoder'].append([adapter_name, task_tgt])
             elif adapter_config['ids'] == 'GROUP':
                 adapter_config['ids'] = list(tgt_groups)
                 for task_key, task_config in opts.in_config[0]['tasks'].items():
                     task_src, task_tgt = task_config['src_tgt'].split('-')
                     task_config['adapters']['decoder'].append([adapter_name, cc_opts['groups'][task_tgt]])
             elif adapter_config['ids'] == 'FULL':
                 adapter_config['ids'] = ['full']
                 for task_key, task_config in opts.in_config[0]['tasks'].items():
                     task_config['adapters']['decoder'].append([adapter_name, 'full'])
                     
     # Write back expanded adapter ID lists under the global 'adapters' section.
     opts.in_config[0]['adapters']['encoder'] = encoder_adapters
     opts.in_config[0]['adapters']['decoder'] = decoder_adapters

     duration = time.time() - start
     logger.info(f'step took {duration} s')


# PURPOSE: Translate per-task adapter selections into layer-indexed stacks (encoder/decoder).
def _adapters_to_stacks(task_adapters, opts, side):
     # Get the global adapter specs (encoder/decoder).
     adapter_specs = opts.in_config[0]['adapters']
     
     # Prepare a list of lists: one inner list per layer on the chosen side.
     adapters = [list() for _ in range(len(opts.in_config[0][f'{side}_layers']))]
     
     # For each (adapter_group, sub_id) pair selected for this task...
     for adapter_group, sub_id in task_adapters:
         
         # Find which layer stack index this adapter group is attached to.
         layer_stack_index = adapter_specs[f'{side}oder'][adapter_group]['layer_stack_index']
         
         # Append the (group, id) to that layer's list.
         adapters[layer_stack_index].append([adapter_group, sub_id])
         
     # Return the per-layer adapter arrangement.
     return adapters

from .schema import register_schema, register_command_io

def register(subparsers):
    p = subparsers.add_parser(
        "adapter_config",
        help="Expand adapter ID spaces (LANGUAGE/GROUP/FULL) and attach per-task selections.",
        description=(
             "Expands adapter id-spaces (LANGUAGE/GROUP/FULL) using config_config.groups and "
             "generates adapter mappings for tasks. Writes adapter sections back to YAML."),
    )
    p.add_argument("--in_config", required=True, type=load_yaml, metavar="FILE.yaml")
    p.add_argument("--out_config", metavar="FILE.yaml")
    p.add_argument("--dry-run", action="store_true", help="Compute but do not write changes.")         
    p.set_defaults(handler=adapter_config)
    p.set_defaults(_mutates_yaml=True)

    register_command_io("adapter_config", {
        "reads": ["tasks", "adapters", "config_config.groups"],
        "writes": ["adapters", "tasks"],
        "summary": "Expands adapter id-spaces (LANGUAGE/GROUP/FULL) and maps tasks to adapters."
    })
    register_command_template_extras("adapter_config", {
         "_notes": [
              "Define adapters under 'adapters'. You can refer to them in tasks by name.",
              "Example shows a shared adapter and a language-specific one."
         ],
         "tasks": {},
         "adapters": {
              "shared": {"type": "bottleneck", "dim": 256},
              "lang_en": {"type": "bottleneck", "dim": 256}
         },
         "config_config": {
              "groups": {"en": 0, "fi": 1}
         }
    })
    
