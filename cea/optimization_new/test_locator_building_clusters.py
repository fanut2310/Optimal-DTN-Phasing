from cea.inputlocator import InputLocator
locator = InputLocator(scenario_path)
print(locator.get_building_clusters())  # Should print valid path