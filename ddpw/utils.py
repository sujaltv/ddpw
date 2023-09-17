from typing import Optional


class Utils:
    verbose: Optional[bool] = True
    r"""A global boolean property that specifies if the wrapper must print
    log contents to the console or not."""
  
    @staticmethod
    def print(*args, **kwargs):
        r"""
        A custom print wrapper that prints the contents if the process is
        running in the verbose mode (`i.e.`, ``verbose = True``). This method is
        a simple check around Python's system print function.
  
        :param bool verbose: To print or not to print. Default: ``None``.
        """
  
        if 'flush' not in kwargs:
            kwargs['flush'] = True
  
        if kwargs.get('verbose', Utils.verbose):
            if 'verbose' in kwargs: del(kwargs['verbose'])
            print(*args, **kwargs)
  
