FROM ubuntu:16.04

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

# sox i
RUN apt-get install sox -y
RUN pip3 install sox
RUN apt-get install python3-numpy python3-scipy python3-nose -y
RUN apt-get install gfortran libopenblas-dev liblapack-dev -y

# main packs
RUN apt-get install libav-tools -y
RUN pip3 install theano keras librosa h5py scikit-image boto

WORKDIR /usr/src/

COPY . .
COPY configs/keras.json /root/.keras/
COPY configs/.theanorc /root/.theanorc
#RUN cat ~/.keras/keras.json
#RUN apt show libomp-dev
#RUN python3 -c 'import theano; print(theano.config.openmp)'
#RUN OMP_NUM_THREADS=8 python3 app.py
CMD OMP_NUM_THREADS=36 python3 app.py

#ENTRYPOINT ["python3"]