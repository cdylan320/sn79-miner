# Simple mock for ypyjson
class YpyObject(dict):
    """Simple mock for YpyObject that behaves like a dictionary"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def __getitem__(self, key):
        return super().__getitem__(key)
        
    def __setitem__(self, key, value):
        return super().__setitem__(key, value)
        
    def get(self, key, default=None):
        return super().get(key, default)
