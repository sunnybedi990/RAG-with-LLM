# Frontend Dockerfile
FROM node:16-alpine

# Set working directory
WORKDIR /app

# Copy the frontend code into the container
COPY . .

# Install dependencies
RUN npm install

# Build the frontend
RUN npm run build

# Expose frontend port
EXPOSE 3000

# Serve the frontend
CMD ["npm", "start"]
