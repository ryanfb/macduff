all: macduff xyzfit

macduff: macduff.cpp colorchecker.h
	g++ -I/opt/local/include -L/opt/local/lib macduff.cpp -lcvaux -lcv -lhighgui -lcxcore -o macduff

xyzfit: xyzfit.cpp colorchecker.h
	g++ -I/opt/local/include -L/opt/local/lib xyzfit.cpp -lcvaux -lcv -lhighgui -lcxcore -o xyzfit