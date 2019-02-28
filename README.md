# mate21
Tópicos em Computação Visual III

Arquivos utilizados durante a disciplina MATE21 do departamento de computação da UFBA.

A pasta ```data_part1``` contém o dataset utilizado neste projeto. A base de treino consiste em 5000 imagens de caracteres numéricos (0 - 9), de tamanho 71 x 77, separados por pastas. A base de teste contém 5490 imagens de caracteres numéricos (0 - 9) em uma única pasta.

Na pasta ```other-files``` contém códigos utilizados na primeira parte da disciplina (sem o tensorflow).

Na pasta ```source``` contém códigos utilizados na segunda parte da disciplina (com o tensorflow).

Os códigos utilizados na terceira parte da disciplina estão organizados em um [outro repositório](https://github.com/moreiralucas/mate21-gan).


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

