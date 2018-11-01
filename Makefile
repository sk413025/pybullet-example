all: runmodel

runmodel:
	mkdir -p output/result
	cp parameters.yaml output
	papermill simple_training_example.ipynb \
				output/result-ppo.ipynb \
				-f output/parameters.yaml \
				-p log_dir ./output/result

