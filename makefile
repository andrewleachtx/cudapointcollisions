build:
	cmake -S . -B build 
	cmake --build build

clean:
	cmake --build build --target clean

# proper syntax is for example, "make run num_particles=1000"
# https://stackoverflow.com/questions/2214575/passing-arguments-to-make-run
# routing build output to dev/null
run:
	cmake --build build > /dev/null
	./build/CUDAPARTICLESYSTEMS $(n)