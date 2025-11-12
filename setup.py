from setuptools import setup, find_packages

setup(
    #o nome que sera usado no pip install
    #vale ressaltar que o nome do pip install eh diferente (!=) do nome do import
    #o nome do import vai ser o nome da pasta que ta dentro de src, no caso CB2325NumericaG1

    name="CB2325-numerica-G1",

    #aqui na versao, a gente vai ir atualizando sempre que a gente botar um novo codigo no PyPI
    #em geral a gente usa o esquema MAJOR.MINOR.PATCH:
        #MAJOR eh que vai mudar muita coisa, e vai gerar incompatibilidade com algum código antigo
        #MINOR eh implementacao de feature nova, não gera imcompatibilidade
        #PATCH eh correcao de bug
    version="1.0.0",
    
    #informacoes sobre a biblioteca
    #esses campos sao opcionais, ai cada um bota seu nome e email (se quiser)
    description="Biblioteca de cálculo numérico produzida pelos alunos do Grupo 1, para a disciplina de Programação 2, do IMPA Tech, ministrada pelo professor Emilio Vital Brazil..",
    author="Alexander Kahleul, Cauan Carlos Rodrigues Dutra, Juan Martins Santos, Luana Fagundes De Lima, Luana Mognon Da Silva, Lucas Fraga Damasceno, Mariana Tiemi Yoshioka, Mateus Stacoviaki Galvão, Micaele Magalhães Brandão Veras, Rafael Augusto De Almeida, Ryan Carvalho Pereira Dos Santos",
    author_email="al.alexander.kahleul@impatech.edu.br, al.cauan.dutra@impatech.edu.br, al.juan.santos@impatech.edu.br, al.luana.lima@impatech.edu.br, al.luana.silva@impatech.edu.br, al.lucas.damasceno@impatech.edu.br, al.mariana.yoshioka@impatech.edu.br, al.mateus.galvao@impatech.edu.br, al.micaele.veras@impatech.edu.br, al.rafael.almeida@impatech.edu.br, al.ryan.santos@impatech.edu.br",
    project_urls={
        #o link do github do projeto
        "Código fonte": "https://github.com/vrsmic/CB2325NumericaG1",
        "Documentação": "https://cb2325numericag1.readthedocs.io/pt-br/latest/",
        
    },

    #pacotes
    package_dir={"": "src"},
    #isso aqui eh obrigatorio. basicamente serve para encontrar as packages dentro do diretório src
    packages=find_packages(where="src"),
    
    #dependencia que utilizamos. se vc usar alguma dependencia, coloca nessa lista
    install_requires=[
        "numpy>=1.20.0", #se algm for usar numpy
        "matplotlib",
        "typing", 
        "sympy",
        "setuptools"
    ],
    
    #requerimentos da versao do python
    python_requires=">=3.10",
)