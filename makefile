build:
	cmake -S . -B build 
	cmake --build build

clean:
	cmake --build build --target clean

# proper syntax is for example, "make run n=1000 threads=512" 
# https://stackoverflow.com/questions/2214575/passing-arguments-to-make-run
# routing build output to dev/null
run:
	cmake --build build
	./build/CUDAPOINTCOLLISIONS $(n) $(threads)