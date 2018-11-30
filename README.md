# mate21
Tópicos em Computação Visual III

Arquivos utilizados durante a disciplina MATE21 do departamento de computação da UFBA.

Na pasta ```other-files``` contém códigos utilizados na primeira parte da disciplina (sem o tensorflow).


Na pasta ```source``` contém códigos utilizados na segunda parte da disciplina (com o tensorflow).


### Instruções para utilizar os códigos

Instale o virtual env:

```
sudo apt install python3-venv
```

Crie e ative com:

```
python3 -m venv env
source env/bin/activate
```
Instale os requerimentos para usar a CPU:

```
pip install -r cpu-requirements.txt
```
ou, para GPU:

```
pip install -r gpu-requirements.txt
```


Mais informações sobre [Tensorflow GPU dependencies](https://www.tensorflow.org/install/gpu)