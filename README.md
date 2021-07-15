# Mitosis Detection Web App

(Open preview of this file using Ctrl+Shift+V)

## Pre-requisites

1. Install Python 3.6.x
2. Install Nodejs

## Setup

1. Create python environment.

    ```cmd
    python -m venv web_app_env
    cd web_app_env
    ```

2. unzip `web_app.zip` in `web_app_env`, this will create folder `web_app` inside `web_app_env`.

3. Install `http-server` node module.

    ```cmd
    npm config set proxy http://username:password@172.30.10.11:3128 
    npm config set https-proxy http://username:password@172.30.10.11:3128 
    npm install -g http-server
    ```

4. Create directory `trained_models` and place models inside it.

5. Create directory `img_dataset` and place test images inside it.

## Run

1. To run front-end, open command prompt in folder `web_app_env/web_app` and run following command:

    ```cmd
    http-server .
    ```

2. To run back-end, open command prompt in folder `web_app_env` and activate python environment:

    ```cmd
    Scripts\activate
    ```

3. Now, run following commands:

    ```cmd
    cd web_app
    set FLASK_APP=app.py
    set FLASK_ENV=development
    python app.py
    ```
