python3 tulip.py tlp.tlp && ./output &&
nasm -felf64 generated.asm &&
ld -o exe generated.o