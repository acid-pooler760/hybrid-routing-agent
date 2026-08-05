from omegaconf import DictConfig, ListConfig, OmegaConf
from typing import Any, List, Tuple


##################################################
#              config utils
##################################################
def get_config():
    from pathlib import Path
    cli_conf = OmegaConf.from_cli()
    yaml_path = Path(cli_conf.config)
    yaml_conf = OmegaConf.load(yaml_path)
    # Handle 'defaults:' manually (plain OmegaConf.load doesn't do Hydra inheritance)
    defaults = yaml_conf.pop("defaults", None)
    base_conf = OmegaConf.create({})
    if defaults is not None:
        for entry in defaults:
            ref = str(entry).strip()  # e.g. '../osworld_rl'
            ref_path = (yaml_path.resolve().parent / ref).with_suffix(".yaml")
            if ref_path.exists():
                base_conf = OmegaConf.merge(base_conf, OmegaConf.load(ref_path))
    conf = OmegaConf.merge(base_conf, yaml_conf, cli_conf)
    return conf


def flatten_omega_conf(cfg: Any, resolve: bool = False) -> List[Tuple[str, Any]]:
    ret = []

    def handle_dict(key: Any, value: Any, resolve: bool) -> List[Tuple[str, Any]]:
        return [(f"{key}.{k1}", v1) for k1, v1 in flatten_omega_conf(value, resolve=resolve)]

    def handle_list(key: Any, value: Any, resolve: bool) -> List[Tuple[str, Any]]:
        return [(f"{key}.{idx}", v1) for idx, v1 in flatten_omega_conf(value, resolve=resolve)]

    if isinstance(cfg, DictConfig):
        for k, v in cfg.items_ex(resolve=resolve):
            if isinstance(v, DictConfig):
                ret.extend(handle_dict(k, v, resolve=resolve))
            elif isinstance(v, ListConfig):
                ret.extend(handle_list(k, v, resolve=resolve))
            else:
                ret.append((str(k), v))
    elif isinstance(cfg, ListConfig):
        for idx, v in enumerate(cfg._iter_ex(resolve=resolve)):
            if isinstance(v, DictConfig):
                ret.extend(handle_dict(idx, v, resolve=resolve))
            elif isinstance(v, ListConfig):
                ret.extend(handle_list(idx, v, resolve=resolve))
            else:
                ret.append((str(idx), v))
    else:
        assert False

    return ret
