class Registry:
    """
    A class for registering and retrieving various modules or components.

    This can be used to manage different types of modules such as datasets, models, encoders, etc., in a centralized registry.

    Attributes:
    - _modules (dict): A private dictionary to store the registered modules.
    """

    def __init__(self):
        """Initialize the registry with an empty dictionary."""
        self._modules = {}


    def register_module(self, name=None,module=None):
        """
        Register a module with a specific name.

        Parameters:
        - name (str, optional): The name with which the module will be registered. If None, the module's __name__ attribute will be used.
        - module: The module to be registered.

        Returns:
        - The module that was registered.

        Raises:
        - AssertionError: If a module with the given name is already registered.
        """
        def _register_module(module):
            key = name
            if key is None:
                key = module.__name__
            assert key not in self._modules,f"{key} is already registered."
            self._modules[key]=module
            return module

        if module is not None:
            return _register_module(module)

        return _register_module

    def get(self,name):
        """
        Retrieve a registered module by its name.

        Parameters:
        - name (str): The name of the module to retrieve.

        Returns:
        - The requested module.

        Raises:
        - AssertionError: If no module with the given name is registered.
        """
        assert name in self._modules,f"{name} is not registered."
        return self._modules[name]


def build_from_cfg(cfg,registry,**kwargs):
    """
    Build an object from a configuration and a registry.

    Parameters:
    - cfg (str, dict, list, or None): The configuration for building the object. It can be a string (name of the registered module), a dictionary (containing the type and other arguments for the object), a list (of configurations), or None.
    - registry (Registry): The registry from which to retrieve the module.
    - **kwargs: Additional keyword arguments.

    Returns:
    - The built object based on the provided configuration.

    Raises:
    - TypeError: If the configuration type is not supported or if there is an error in initializing the object.
    """

    if isinstance(cfg,str):
        return registry.get(cfg)(**kwargs)
    elif isinstance(cfg,dict):
        args = cfg.copy()
        args.update(kwargs)
        obj_type = args.pop('type')
        obj_cls = registry.get(obj_type)
        try:
            module = obj_cls(**args)
        except TypeError as e:
            if "<class" not in str(e):
                e = f"{obj_cls}.{e}"
            raise TypeError(e)

        return module
    elif isinstance(cfg,list):
        from jittor import nn
        return nn.Sequential([build_from_cfg(c,registry,**kwargs) for c in cfg])
    elif cfg is None:
        return None
    else:
        raise TypeError(f"type {type(cfg)} not support")


# Initialize various registries
DATASETS = Registry()
MODELS = Registry()
ENCODERS = Registry()
DECOMPOSERS = Registry()
DIFFUSIONS = Registry()
NETS = Registry()
SCHEDULERS = Registry()
HOOKS = Registry()
LOSSES = Registry()
OCCUPANCIES = Registry()
OPTIMS = Registry()
SAMPLERS = Registry()
METRICS = Registry()
SEGMENTORS = Registry()
GENERATORS = Registry()
DISCRIMINATORS = Registry()