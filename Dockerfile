FROM python:3.8

WORKDIR /code

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY script.py .

COPY script_2.py .

COPY dataset/ /code/dataset/

CMD [ "python", "script.py" ]
#CMD [ "python", "script_2.py" ]
