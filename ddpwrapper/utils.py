def add_click_options(options):
  def _add_click_options(func):
    for option in reversed(options):
      func = option(func)
    return func
  return _add_click_options
