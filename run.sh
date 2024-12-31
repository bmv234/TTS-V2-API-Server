#!/bin/bash

# Generate self-signed certificate and private key
openssl req -x509 -newkey rsa:4096 -nodes -keyout key.pem -out cert.pem -days 365 -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"

# Make the script executable
chmod +x main-ui.py

# Run the server with HTTPS
python main-ui.py
