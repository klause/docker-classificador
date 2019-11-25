# Deploy do seu modelo fastai no Heroku (Web App)

## Agradecimento

[Texto de introdução](https://github.com/piegu/glasses-or-not/edit/master/README.md) e [tutorial](https://github.com/piegu/glasses-or-not/blob/master/tutorial/Web_app_fastai.md)...
- (muito, muito, muito) inspirado do [trabalho](https://github.com/weltonrodrigo/fastai-v3) do [Welton Rodrigo Torres Nascimento](https://github.com/weltonrodrigo) (até cópias/colas de seus textos porque não fazia sentido modificá-los)
- (muito, muito) inspirado do [trabalho](https://github.com/render-examples/fastai-v3) do [Anurag Goel](https://github.com/anurag)
- (muito) inspirado do [trabalho](https://github.com/nikhilno1/healthy-or-not/blob/master/heroku-deploy.md) do [Nikhil](https://github.com/nikhilno1)

Muito obrigado a eles!

## Colocar na Internet o seu modelo baseado em fastai 

Existem várias formas de colocar na Internet o seu modelo baseado em fastai. A página de informações do curso mostra diversas opções:

* [Render](https://course.fast.ai/deployment_render.html)
* [Google App Engine](https://course.fast.ai/deployment_google_app_engine.html)
* [AWS Lambda](https://course.fast.ai/deployment_aws_lambda.html)
* [Amazon SageMaker](https://course.fast.ai/deployment_amzn_sagemaker.html)
* [AWS BeanStalk](https://course.fast.ai/deployment_aws_beanstalk.html)
* [Microsoft Azure Functions](https://course.fast.ai/deployment_azure_functions.html)

Com certeza, uma opção que se destaca é o Render.com ([tutorial online](https://course.fast.ai/deployment_render.html)), 
que é uma plataforma criada pelo mesmo criador do [Crestle](https://crestle.ai/) mas uma outra opçao é usar gratuitamente o Heroku.

## 1) Exporte o modelo treinado

Gere o seu modelo normalmente e salve o `export.pkl` com `learn.export()`.

Coloque que arquivo no google drive ou dropbox (porque é de graça). É necessário gerar um link para o arquivo usando um dos dois serviços abaixo:

- Google Drive: Use [este gerador de links](https://www.wonderplugin.com/online-tools/google-drive-direct-link-generator/).
- Dropbox: Use [este gerador de links](https://syncwithtech.blogspot.com/p/direct-download-link-generator.html).

## 2) Ajuste a aplicação pré-pronta

Faça um fork no github deste repositório base, que contém uma aplicação pré-pronta e customizável.

Edite, no próprio Github, o arquivo [app/server.py](../app/server.py) para incluir:

1. O endereço do `export.pkl` em `export_file_url` ou o ID do arquivo no Google Drive em `export_file_google_drive_id` caso o arquivo seja maior que 100MB (limite do scanner do antivitus).
2. Troque o nome das classes geradas pelo seu modelo em `classes = ['Direito', 'Economia', 'Social']`.
3. Lembre-se de usar a opção "Commit directly to the master branch".
4. Edite também o texto que aparece para os usuários da sua aplicação, no arquivo [app/static/index.html](../app/static/index.html):
    1. O título (tag `<title>`)
    2. A descrição (`meta name="description"`)
5. E pronto!

## 3) Deploy no Heroku

Apesar de ser gratuito, o deploy no Heroku exige passos adicionais, que devem ser repetidos manualmente **sempre que o código da aplicação for alterado na Github**.

Normalmente o deploy no Heroku exige apenas o código da aplicação, mas, no caso dos modelos do fastai, ela é grande demais para o método tradicional e exige o **deploy em containers docker**. Para isso, é necessário executar os comandos numa máquina com docker instalado.

Se estiver usando uma máquina do Google Cloud para gerar o modelo, pode-se usar o Cloud Shell para fazer o deploy. Se esse for o caso, execute os passos do item 3.1.

Se estiver em uma máquina local Linux, pule para o passo 3.2.

### 3.1) Acesse o console do Google Cloud

Os passos a seguir consideram que você já tem uma conta no gcloud.

1. Acesse o dashboard em [https://console.cloud.google.com/home/dashboard](https://console.cloud.google.com/home/dashboard).
2. Abra o Cloud Shell no topo da página:

    ![](Captura_de_Tela_2019-04-10_as_13-6bc61c51-8907-4603-9c7c-df9ac066361f.33.19.png)

3. Ative o modo de otimização. Ele faz com que o Cloud Shell rode numa máquina mais potente e permita o build do container.

    ![](Captura_de_Tela_2019-04-10_as_13-306f313f-7622-4f88-bf4c-e0aebf9eda2a.36.17.png)


### 3.2) Usando o cliente da heroku para deploy

1. Instale o cliente da heroku:

    `curl https://cli-assets.heroku.com/install.sh | sh`

    ![](0d2b036e-4920-42c7-83a9-7dc99d8b8966.jpg)

1. Faça o clone da aplicação modificada e entre no diretório. Note que o seu endereço vai ser diferente do meu.

    `git clone https://github.com/klause/classificador-consultoria.git`

    ![](60804939-9189-476e-a74f-9c2335411b7e.jpg)
    
6. Entre na pasta do seu modelo. Note que o nome da sua pasta vai ser diferente do meu.

    `cd classificador-consultoria`

    ![](7d12a72b-13cc-406c-a5b5-5378166e3730.jpg)

7. Faça o login no heroku:

    `heroku login`
    
    O cliente heroku vai pedir que você aperte enter e depois apresentar um link para ser aberto na internet. Esse link vai te levar para a página de login. Nessa tela você pode fazer seu cadastro se ainda não tiver. Depois da autenticação bem sucedida, o cliente vai informar que está logado.

    ![](48b7d437-7d51-4a8f-b69f-8f30bd14f842.jpg)

8. Crie um novo app no Heroku para receber sua aplicação:

    `heroku create -a <app-name>`

    O nome da aplicação também vai ser o endereço dela na Internet. No exemplo acima, seria nome-da-aplicacao.app.herokuapp.com. Caso você não queira um nome específico, ele vai criar um nome aleatório.

    ![](493dd0b5-1c3e-4434-9971-3669313ffe02.jpg)

9. É necessário informar ao cliente docker como fazer o login no heroku, mas o próprio cliente toma conta disso:

    `heroku container:login`

    ![](5cccb037-80b0-4b33-8015-1b151460848f.jpg)

10. Agora para construir e enviar o container para o heroku. Isso deve levar em torno de 5 minutos. O print abaixo mostra só o começo do texto de saída gerado pelo comando.

    `heroku container:push web -a <app-name>`

    ![](ead96a7f-8ca6-4a96-ae52-d0335ffb1ab0.jpg)

    Se a construção e envio forem bem sucedidos, deve gerar uma saída parecida com esta:

    ![](4adc42d4-7955-4dc4-863c-f77900147b72.jpg)

11. Agora é necessário informar o keroku que faça o deploy propriamente dito do container criado:

    `heroku container:release web -a <app-name>`

    ![](bac4993a-6071-4527-9278-71e6ee9188b8.jpg)

12. Acesse seu aplicativo.

    `heroku open`
    
    Se isso não abrir a janela do navegador, seu aplicativo já estará disponível em https://<nome_da_aplicação>.herokuapp.com. No nosso exemplo em https://classificador-consultoria.herokuapp.com/
