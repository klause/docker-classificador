import asyncio
import uvicorn
from gdrive_download import download_file_from_google_drive
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

#export_file_url = 'https://drive.google.com/uc?export=download&id=1EQBlYB7cUKoVc1SABB65McnPzjrQsNsX'
export_file_url = None

# download of big files in google drive needs a confirmation step, set this variable instead of url
export_file_google_drive_id="1EQBlYB7cUKoVc1SABB65McnPzjrQsNsX"

export_file_name = 'export.pkl'

classes = ['Direito', 'Economia', 'Social']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file():
    dest = path / export_file_name
    if dest.exists(): return
    if export_file_url is not None:
        async with aiohttp.ClientSession() as session:
            async with session.get(export_file_url) as response:
                data = await response.read()
                with open(dest, 'wb') as f:
                    f.write(data)
    else:
        download_file_from_google_drive(export_file_google_drive_id, path / export_file_name)

async def setup_learner():
    await download_file()
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'static' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/predict', methods=['POST'])
async def analyze(request):
    body = await request.body()
    text_data = body.decode()

    prediction = learn.predict(text_data)

    idx_classe = prediction[1].item()

    print(str(prediction))

    probs = [{ 'classe': classes[i], 'probabilidade': prediction[2][i].item() } for i in range(len(prediction[2]))]

    result = {
        'idx_classe': idx_classe,
        'nome_classe': classes[idx_classe],
        'probabilidade': prediction[2][idx_classe].item(),
        'lista_prob': probs
    }
    return JSONResponse({'result': result})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        port = int(os.getenv('PORT', 5042))
        uvicorn.run(app=app, host='0.0.0.0', port=port)
