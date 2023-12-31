import time
import asyncio
from twitchio.ext import commands
import pandas as pd
from unidecode import unidecode
import random
import nltk
from nltk.tokenize import word_tokenize
import pydirectinput

nltk.download('punkt')  # Descargar el tokenizador

todos_comentarios = []

# Cargar el archivo CSV en un DataFrame
file_path = 'Detect_object\Script 1\Groser-IAs.csv' # Asegúrate de dar la ruta correcta al archivo
df = pd.read_csv(file_path, encoding="ISO-8859-1",skiprows=1)  # Saltar la primera fila adicional

# Función para reemplazar números y símbolos por letras
def replace_numbers_and_symbols(text):
    replacements = {
        '0': 'o',
        '1': 'l',
        '3': 'e',
        '4': 'a',
        '5': 's',
        '7': 't',
        '8': 'b',
        '@': 'a',
        '$': 's',
        '&': 'y'
    }
    for num, letter in replacements.items():
        text = text.replace(num, letter)
    return text


#------------- COMMANDS and KEYS to respond a certain event -------------

COMMANDS_AND_KEYS: list[(str, str)] = [
    ("Select", "z"),
    ("Start", "x"),
    ("Arriba", "up"),
    ("Abajo", "down"),
    ("Izquierda", "left"),
    ("Derecha", "right"),
    ("A", "a"),
    ("B", "b"),
    ("R", "r"),
    ("L", "l"),
    
    ("select", "z"),
    ("start", "x"),
    ("arriba", "up"),
    ("abajo", "down"),
    ("izquierda", "left"),
    ("derecha", "right"),
    ("a", "a"),
    ("b", "b"),
    ("r", "r"),
    ("l", "l"),
]


def _send_key(key):
    """
    Simulate a key press using pydirectinput.
    """
    pydirectinput.keyDown(key)
    time.sleep(0.1)  # Adjust the delay as needed
    pydirectinput.keyUp(key)


class Tchbot(commands.Bot):

    def __init__(self, token, channel):
        super().__init__(token=token, prefix='!', initial_channels=[channel])
        self.last_message_time = time.time()

    async def event_ready(self):
        print(f'Bot listo como {self.nick}')
        
    async def execute_command_action(self, command):
        key_mapping = dict(COMMANDS_AND_KEYS)
        key = key_mapping.get(command)
        
        if key:
            _send_key(key)
            #await asyncio.sleep(0.1)  # Adjust the delay as needed

    async def event_message(self, message):
        if message.author is None:
            return
        try:

            self.last_message_time = time.time()

            content = unidecode(message.content.lower())  # Convertir a minúsculas y quitar acentos
            tokens = word_tokenize(content)
            content = ' '.join([replace_numbers_and_symbols(token) for token in tokens])
            print(content)

            # Para etiquetar al usuario en la respuesta
            usuario = f"@{message.author.name}" if message.author.name else "Usuario Desconocido"

            # Check if the message content matches a command
            content_lower = content.lower()
            for command, _ in COMMANDS_AND_KEYS:
                if command.lower() in content_lower:
                    await self.execute_command_action(command.lower())
                    break
            
            # Buscar en el DataFrame para palabras clave y categorías
            matched_rows = df[df['Pregunta'].str.contains(content, case=False, na=False)]

            if not matched_rows.empty:
                # Seleccionar una fila al azar que coincida
                selected_row = random.choice(matched_rows.index.tolist())

                # Seleccionar una respuesta al azar de las disponibles en la fila seleccionada
                available_responses = matched_rows.loc[selected_row, 'Respuesta':].dropna().tolist()
                random_response = random.choice(available_responses)

                await message.channel.send(f"{usuario} {random_response}")

        except Exception as e:
            print(f"Ocurrió un error: {e}")

        # Check the time difference
        if time.time() - self.last_message_time > 60:  # 60 seconds = 1 minute
            # Send the "99, 99" message
            await message.channel.send("99, 99")

        await self.handle_commands(message)
        
    @commands.command(name='Select')
    async def select1_command(self, ctx):
        await ctx.send(f'Tecla Select detectada!')
        
    @commands.command(name='select')
    async def select2_command(self, ctx):
        await ctx.send(f'Tecla Select detectada!')

    @commands.command(name='Start')
    async def start1_command(self, ctx):
        await ctx.send(f'Tecla Start detectada!')
    
    @commands.command(name='start')
    async def start2_command(self, ctx):
        await ctx.send(f'Tecla Start detectada!')
        
    @commands.command(name='Arriba')
    async def up1_command(self, ctx):
        await ctx.send(f'¡Movimiento hacia arriba detectado!')
        
    @commands.command(name='arriba')
    async def up2_command(self, ctx):
        await ctx.send(f'¡Movimiento hacia arriba detectado!')
        
    @commands.command(name='Abajo')
    async def down1_command(self, ctx):
        await ctx.send(f'¡Movimiento hacia abajo detectado!')

    @commands.command(name='abajo')
    async def down2_command(self, ctx):
        await ctx.send(f'¡Movimiento hacia abajo detectado!')
        
    @commands.command(name='Izquierda')
    async def left1_command(self, ctx):
        await ctx.send(f'¡Movimiento hacia la izquierda detectado!')

    @commands.command(name='izquierda')
    async def left2_command(self, ctx):
        await ctx.send(f'¡Movimiento hacia la izquierda detectado!')

    @commands.command(name='Derecha')
    async def right1_command(self, ctx):
        await ctx.send(f'¡Movimiento hacia la derecha detectado!')

    @commands.command(name='derecha')
    async def right2_command(self, ctx):
        await ctx.send(f'¡Movimiento hacia la derecha detectado!')

    @commands.command(name='A')
    async def a1_command(self, ctx):
        await ctx.send(f'Tecla A detectada!')
        
    @commands.command(name='a')
    async def a2_command(self, ctx):
        await ctx.send(f'Tecla A detectada!')
        
    @commands.command(name='B')
    async def b1_command(self, ctx):
        await ctx.send(f'Tecla B detectada!')
        
    @commands.command(name='b')
    async def b2_command(self, ctx):
        await ctx.send(f'Tecla B detectada!')
        
    @commands.command(name='R')
    async def r1_command(self, ctx):
        await ctx.send(f'Tecla R detectada!')

    @commands.command(name='r')
    async def r2_command(self, ctx):
        await ctx.send(f'Tecla R detectada!')
        
    @commands.command(name='L')
    async def l1_command(self, ctx):
        await ctx.send(f'Tecla L detectada!')
        
    @commands.command(name='l')
    async def l2_command(self, ctx):
        await ctx.send(f'Tecla L detectada!')

    def run_tchbot(self, token, channel):
        bot = Tchbot(token, channel)
        bot.start()