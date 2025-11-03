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
    version="0.1.7",
    
    #informacoes sobre a biblioteca
    #esses campos sao opcionais, ai cada um bota seu nome e email (se quiser)
    description="Uma biblioteca de cálculo numérico (teste).",
    author="Lucas F. Damasceno, ...",
    author_email="damasceno.lucas512@gmail.com, ...",
    project_urls={
        #o link do github do projeto
        "Código fonte": "https://github.com/vrsmic/CB2325NumericaG1",
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