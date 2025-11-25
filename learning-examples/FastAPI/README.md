# Introduction to FastAPI

FastAPI is a modern web framework for building APIs with Python.
It’s designed to be fast, easy to use, and ideal for machine-learning applications.

Many traditional Python web frameworks (like Flask or Django) work well, but FastAPI makes a few things much easier:

- **Very fast performance**: Built on top of high-performance ASGI servers like Uvicorn, FastAPI is one of the fastest Python frameworks available.

- **Automatic validation**: You define the input and output shapes using Python types, and FastAPI automatically checks incoming data for you.

- **Interactive API docs for free**: When you run a FastAPI app, it automatically generates a sleek, interactive documentation page using Swagger UI `http://localhost:8000/docs`. This makes testing your API extremely easy.

- **Great for ML models**: FastAPI is widely used to deploy machine-learning models because it:

    - handles JSON and file uploads cleanly,
    - works nicely with Python data science libraries,
    - and keeps the API code very small and readable.
