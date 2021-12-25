python3 tulip.py tlp.tlp
./output
nasm -felf64 tlp_gen.asm
ld -o exe tlp_gen.o