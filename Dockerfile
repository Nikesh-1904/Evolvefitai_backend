# Step 1: Use an official Python runtime as the base image
# We specify python:3.11 to match your local environment
FROM python:3.11-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Set environment variables to prevent Python from buffering output
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Step 4: Copy only the requirements file first to leverage Docker's caching
COPY requirements.txt .

# Step 5: Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Copy the rest of your application code into the container
COPY . .

# Step 7: Run your Alembic database migrations
# This ensures your database is up-to-date before the app starts
RUN alembic upgrade head

# Step 8: Expose the port the app runs on
EXPOSE 8000

# Step 9: The command to run your application
# Railway will automatically map its internal port to this
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]