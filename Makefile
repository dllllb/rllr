JUPCLEAR="jupyter nbconvert --clear-output --inplace"

.PHONY: project.zip

project.zip: code
	# prepare a temporary directory
	$(eval target := $(shell mktemp -d))

	# propagate the Makefile, the readme, and the license
	cp ./Makefile "${target}"/
	cp ./README.md "${target}"/
	# cp ./LICENSE "${target}"/

	# copy notebooks and clean up
	if [ -d './experiments' ]; then \
		cp -r ./experiments/. "${target}"; \
	fi
		# find "${target}" -type f -name "*.ipynb" -exec "${JUPCLEAR}" {} \; ; \
		# find "${target}" -type f ! -name "*.ipynb" -delete; \

	# put a sdist of the package into the folder
	# cp ./dist/*.tar.gz "${target}"/
	cp -p "`ls -dtr1 ./dist/*.tar.gz | tail -1`" "${target}"/

	# zip it all and cleanup
	$(eval zip := $(shell mktemp))
	cd "${target}"; \
		zip -9qr "${zip}.zip" . -x ".*" -x "__MACOSX" -x ".ipynb_checkpoints"

	mv "${zip}.zip" ./project.zip

	rm -rf "${target}"

code:
	python setup.py sdist

all: project.zip
