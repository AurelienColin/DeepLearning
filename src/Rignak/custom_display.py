# Mock implementation for Rignak.custom_display

class Display:
    def __init__(self, *args, **kwargs):
        """Mock Display class."""
        # Simulate a basic structure that can be indexed if needed by plotter
        self._subplots = {} 

    def __getitem__(self, item):
        """Allow subplot access like display[index]."""
        # Return a mock subplot or self if that's how it's used
        # For simplicity, returning self, assuming methods are called on it.
        # If subplots have their own methods, a mock subplot class would be needed.
        if item not in self._subplots:
            # Create a mock subplot on demand
            self._subplots[item] = MockSubplot()
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
