# Build step #1: build the React frontend
FROM node:14-alpine as build-step
WORKDIR /app
COPY package*.json ./
COPY ./frontend ./frontend
RUN cd frontend && npm install && npm run build --max-old-space-size=420

# Build step #2: build the Flask backend with the frontend as static files
FROM python:3.9
WORKDIR /app
COPY --from=build-step /app/frontend/build ./frontend/build

COPY ./backend ./backend
RUN apt-get update && apt-get install -y python3 python3-pip
RUN cd backend && pip3 install -r requirements.txt

# Expose the port on which the Flask app will run (10000 for Render)
EXPOSE 10000

# Set the command to run the Flask app
CMD ["python3", "backend/application.py"]