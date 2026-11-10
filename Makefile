install:
	python -m pip install -r requirements.txt

run:
	streamlit run app.py

docker:
	docker compose up --build
