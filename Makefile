all: macduff xyzfit applyfit

macduff: macduff.cpp colorchecker.h
	g++ -O3 -I/opt/local/include -L/opt/local/lib macduff.cpp -lcvaux -lcv -lhighgui -lcxcore -o macduff

xyzfit: xyzfit.cpp colorchecker.h
	g++ -g -O3 -I/opt/local/include -L/opt/local/lib xyzfit.cpp -lcvaux -lcv -lhighgui -lcxcore -lgsl -lcblas -latlas -o xyzfit

applyfit: applyfit.cpp colorchecker.h
	g++ -O3 -I/opt/local/include -L/opt/local/lib applyfit.cpp -lcvaux -lcv -lhighgui -lcxcore -o applyfit
