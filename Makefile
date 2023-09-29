install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

	wget https://huggingface.co/datasets/ArtifactAI/arxiv-math-instruct-50k/resolve/main/arxiv_math_instruct_50k.jsonl
	wget https://huggingface.co/datasets/ArtifactAI/arxiv-physics-instruct-tune-30k/resolve/main/arxiv_physics_instruct_30k.jsonl

test:
	python -m pytest -vv test_*.py
#	python -m pytest -vv --cov=ft_llama2 test_*.py

format:	
	black *.py 

lint:
	pylint --disable=R,C --ignore-patterns=test_.*?py *.py

container-lint:
	docker run --rm -i hadolint/hadolint < Dockerfile

refactor: format lint

deploy:
	#deploy goes here
		
all: install lint test format deploy
