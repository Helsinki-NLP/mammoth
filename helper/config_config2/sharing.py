from .utils import load_yaml

# PURPOSE: For each task, compute enc/dec sharing group labels per layer from language/group macros.
def sharing_groups(opts):
    if getattr(opts, "yaml_help", False):
        print_command_yaml_help("sharing_groups")
        
    # Start timing.
    start = time.time()

    cfg = opts.in_config[0]
    cc  = cfg["config_config"]
    
    # Require groups mapping in YAML
    if "groups" not in cc or not isinstance(cc["groups"], dict):
        raise UserConfigError("Missing 'config_config.groups' mapping. Run cluster_languages or define it in YAML.")
    
    # Determine lists; require presence (either CLI or YAML)
    enc = getattr(opts, "enc_sharing_groups", None) or cc.get("enc_sharing_groups")
    dec = getattr(opts, "dec_sharing_groups", None) or cc.get("dec_sharing_groups")
    if not enc or not dec:
        raise UserConfigError("Provide --enc_sharing_groups/--dec_sharing_groups OR set config_config.{enc,dec}_sharing_groups.")

    # Length must match layer counts
    enc_layers = cfg.get("enc_layers")
    dec_layers = cfg.get("dec_layers")
    if not isinstance(enc_layers, list) or not isinstance(dec_layers, list):
        raise UserConfigError("YAML must define enc_layers and dec_layers lists.")
    if len(enc) != len(enc_layers) or len(dec) != len(dec_layers):
        raise UserConfigError("Length mismatch: number of sharing tokens must equal the number of encoder/decoder layers.")

    cc_opts = cc
    groups = cc_opts['groups']
    # Resolve encoder/decoder sharing declarations from CLI or config.
    enc_sharing_groups = opts.enc_sharing_groups if opts.enc_sharing_groups else cc_opts['enc_sharing_groups']
    dec_sharing_groups = opts.dec_sharing_groups if opts.dec_sharing_groups else cc_opts['dec_sharing_groups']
    # Validate presence and per-layer lengths.
    if not enc_sharing_groups:
        raise Exception('Must set --enc_sharing_groups')
    if not dec_sharing_groups:
        raise Exception('Must set --dec_sharing_groups')
    assert len(enc_sharing_groups) == len(opts.in_config[0]['enc_layers'])
    assert len(dec_sharing_groups) == len(opts.in_config[0]['dec_layers'])
    # For each task, translate macros to concrete group names.
    for cname, corpus in opts.in_config[0]['tasks'].items():
        src, tgt = corpus['src_tgt'].split('-')
        mapping_src = {
            'LANGUAGE': src,
            'GROUP': groups[src],
            'FULL': 'full',
            'SRC_LANGUAGE': src,
            'SRC_GROUP': groups[src],
            'TGT_LANGUAGE': tgt,
            'TGT_GROUP': groups[tgt],
        }
        mapping_tgt = {
            'LANGUAGE': tgt,
            'GROUP': groups[tgt],
            'FULL': 'full',
            'SRC_LANGUAGE': src,
            'SRC_GROUP': groups[src],
            'TGT_LANGUAGE': tgt,
            'TGT_GROUP': groups[tgt],
        }
        corpus['enc_sharing_group'] = [mapping_src[sg] for sg in enc_sharing_groups]
        corpus['dec_sharing_group'] = [mapping_tgt[sg] for sg in dec_sharing_groups]

    duration = time.time() - start
    logger.info(f'step took {duration} s')

from .schema import print_schema, COMMAND_IO

def _yaml_help_for_command(cmd):
    keys = ( COMMAND_IO.get(cmd, {}).get("reads", []) +
             COMMAND_IO.get(cmd, {}).get("writes", []) )
    print_schema(sorted(set(keys)))

def register(subparsers):
    p = subparsers.add_parser(
        "sharing_groups",
        help="Resolve encoder/decoder sharing groups per layer (LANGUAGE/GROUP/FULL; allow SRC_/TGT_ prefixes).",
        description=(
            "Produces per-layer lists for encoder/decoder (lengths must match enc_layers/dec_layers). "
            "Requires a 'groups' mapping {lang: group} (from clustering or predefined in YAML)."
        ),
    )
    p.add_argument("--in_config", required=True, type=load_yaml, metavar="FILE.yaml")
    p.add_argument("--out_config", metavar="FILE.yaml")
    p.add_argument("--enc_sharing_groups", action="append", metavar="TOKEN",
        help=('list of {LANGUAGE | GROUP | FULL}. '
              'The first two may be prefixed, e.g. SRC_LANGUAGE or TGT_GROUP. '
              'Default prefix for encoder is SRC.'))
    p.add_argument("--dec_sharing_groups", action="append", metavar="TOKEN",
        help=('list of {LANGUAGE | GROUP | FULL}.'
              'The first two may be prefixed, e.g. SRC_LANGUAGE or TGT_GROUP. '
              'Default prefix for decoder is TGT.'))
    p.add_argument("--yaml-help", action="store_true",
                   help="Show YAML keys this command reads/writes and exit.")
    p.set_defaults(handler=sharing_groups)

