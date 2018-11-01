name := $(shell date +%F-%T)

all: runmodel

runmodel:
	
	@echo ${name}
	mkdir -p eval/${name}
	cp parameters.yaml eval/${name}
	papermill simple_training_example.ipynb \
				eval/ppo-$(shell date +%F-%T).ipynb \
				-f eval/${name}/parameters.yaml \
				-p log_dir eval/${name}

