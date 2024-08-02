# Find-My-Pet

## Table of Contents
1. [Project Overview](#overview)
2. [Motivation](#Motivation)
3. [Data for test](#Data-for-test)
4. [Features](#Features)
5. [Technologies Used](#Technologies-Used)
6. [Installation](#Installation)
   

## Overview
"Find My Pet" is an AI-powered application designed to help reunite lost pets with their owners. When someone finds a lost pet, they can take a picture and upload it to our database. If a pet owner loses their pet, they can upload a previous picture of their pet. The system then matches the uploaded image with the images in the database to find potential matches.

### Motivation
The motivation behind this project is to reunite lost pets with their owners, based on interviews and survey made with pet owners.

## Data for test
   You can get data through this link [Kaggle](https://www.kaggle.com/datasets/andrewmvd/animal-faces)

## Features
- **Image Upload:** Allows users to upload images of found and lost pets.
- **Embeddings Generation:** Converts images to embeddings using a trained ResNet50 model.
- **Matching System:** Matches the embeddings of the uploaded image with those in the database to identify potential matches.

## Technologies Used
- **Backend:** Python, Flask
- **Machine Learning:** ResNet50 for generating image embeddings
- **Database:** Qdrant for storing image embeddings and metadata
- **Deployment:** Server on local_pc

## Installation

### Steps
1. **Clone the repository:**
   ```bash
   git clone https://github.com/omarghazy56/Pet-Finder.git
   cd Pet-Finder

### To make server
1. Install [Node.js](https://nodejs.org/en/download).
2. Install localtunnel globally:
   ```cmd
   npm install -g localtunnel
2. To start the server, use the following command:
   ```cmd
   lt --port 8000 --subdomain your-custom-subdomain
   eg. lt --port 8000 --subdomain abcip