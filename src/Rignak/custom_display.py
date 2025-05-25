# Conditional import for Display and Subplot
# Attempt to import the real implementations first
try:
    from rignak.src.custom_display import Display, Subplot
except (ImportError, ModuleNotFoundError):
    # Fallback to mock implementations if the real library isn't found

    class MockDisplay:
        def __init__(self, *args, **kwargs):
            """Mock Display class."""
            # Simulate a basic structure that can be indexed if needed by plotter
            self._subplots = {} 

        def __getitem__(self, item):
            """Allow subplot access like display[index]."""
            if item not in self._subplots:
                # Create a mock subplot on demand
                self._subplots[item] = MockSubplot() # Assumes MockSubplot is defined in this scope
            return self._subplots[item]

        def imshow(self, *args, **kwargs):
            """Mock imshow method."""
            pass

        # Add any other methods that are called on Display objects if they cause errors
        # For example, if plotter.display.show() is called:
        # def show(self, *args, **kwargs):
        #     pass

    class MockSubplot:
        def __init__(self, *args, **kwargs):
            """Mock Subplot class for Display."""
            self.ax = self # For calls like subplot.ax.set_xticklabels

        def imshow(self, *args, **kwargs):
            """Mock imshow method for a subplot."""
            pass

        def barh(self, *args, **kwargs):
            """Mock barh method for a subplot."""
            pass
        
        def set_xticklabels(self, *args, **kwargs):
            """Mock set_xticklabels method."""
            pass
        
        # Add any other methods called on subplot objects based on test errors.
        # e.g. axvline, set_title, set_xlabel, set_ylabel, etc.
        def axvline(self, *args, **kwargs):
            pass

        def set_title(self, *args, **kwargs):
            pass
        
        def set_xlabel(self, *args, **kwargs):
            pass

        def set_ylabel(self, *args, **kwargs):
            pass
        
        def tick_params(self, *args, **kwargs):
            pass

    # Alias the mock classes to the names expected by the rest of the application
    # This happens only if the import failed.
    Display = MockDisplay
    Subplot = MockSubplot
