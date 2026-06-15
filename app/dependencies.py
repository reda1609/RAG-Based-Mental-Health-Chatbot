# Shared mutable dictionary that holds all loaded models.
# It starts empty and gets populated once during server startup
# inside the lifespan function in main.py.
# Every route reads from here instead of loading models itself.

app_state: dict = {}
