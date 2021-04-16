JUPCLEAR="jupyter nbconvert --clear-output --inplace"

.PHONY: rllr-project.tar.gz

rllr-project.tar.gz: code
	# prepare a temporary directory
	$(eval target := $(shell mktemp -d))

	# propagate the Makefile, the readme, and the license
	cp ./Makefile "${target}"/
	cp ./README.md "${target}"/
	# cp ./LICENSE "${target}"/

	# copy notebooks and unittests
	cp -r ./experiments/. "${target}/experiments"
	cp -r ./tests "${target}/tests"
	# find "${target}/experiments" -type f -name "*.ipynb" -exec "${JUPCLEAR}" {} \;

	# put the latest sdist of the package into the folder
	cp -p "`ls -t1 ./dist/*.tar.gz | head -1`" "${target}"/

	# remove junk
	find "${target}" -type f -name ".*" -delete

	# zip it all and cleanup
	$(eval filename := $(shell mktemp))
	cd "${target}"; \
		tar -zcvf "${filename}.tar.gz" ./
		# zip -9qr "${filename}.zip" . -x ".*" -x "__MACOSX" -x ".ipynb_checkpoints"

	mv "${filename}.tar.gz" "./rllr-project.tar.gz"

	rm -rf "${target}"

code:
	python setup.py sdist

all: rllr-project.tar.gz
