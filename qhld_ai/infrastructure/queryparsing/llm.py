"""LLM query parser.

Binds the ``ParsedQuery`` schema to a chat model via ``with_structured_output`` so
the model returns a validated object, not free text. The chat model is built from
the existing ``create_llm_from_env`` factory on a Settings copy whose ``llm_*`` are
overridden by the decoupled ``query_parser_llm_*`` (empty => fall back to the main
llm settings), so the parser can run on a different model than any future
answer-synthesis. The model is built lazily on first ``parse`` so importing this
module stays cheap.
"""

from datetime import date

from qhld_ai.domain.ports.query_parser import ParsedQuery
from qhld_ai.infrastructure.config.settings import Settings

from .factory import _register

_SYSTEM = """\
Eres un analizador de consultas para un buscador de intervenciones parlamentarias \
del Congreso de los Diputados de España. Dada una consulta en lenguaje natural, \
extrae los filtros estructurados y la consulta semántica residual.

Reglas:
- semantic_query: SOLO el tema o contenido de la intervención (de qué trata), sin \
las restricciones de orador, persona mencionada, grupo/partido ni fechas. Cadena \
vacía si la consulta no tiene tema.
- speakers: TODAS las personas que INTERVIENEN, indicadas por su nombre propio, un \
elemento de la lista por persona (p. ej. 'de Pedro Sánchez y Yolanda Díaz' → \
['Pedro Sánchez', 'Yolanda Díaz']). Null si se refiere a ellas por su cargo, o si \
no se especifica orador.
- mentioned_persons: TODAS las personas que la intervención debe MENCIONAR o \
nombrar, que NO son quien interviene, un elemento por persona (p. ej. 'discursos \
que mencionen a Zapatero y a Rajoy' → ['Zapatero', 'Rajoy']). Null si la consulta \
no pide personas mencionadas.
- mentions_mode: cómo combinar varias personas mencionadas: 'all' si deben \
aparecer todas (conjunción 'y'/'e', o una sola persona: 'que mencionen a Ayuso y \
Putin' → 'all'); 'any' si basta con una (disyunción 'o'/'u': 'que mencionen a \
Ayuso o Putin' → 'any').
- entities: TODAS las entidades nombradas NO personales a las que la intervención \
debe referirse explícitamente — una cosa concreta con nombre propio a la que el \
tema está anclado: una organización ('Navantia', 'la UNRWA'), un evento \
('Eurovisión'), una ley ('la ley de amnistía'), un conflicto ('la guerra de \
Gaza'), un lugar del que TRATA la intervención ('sobre la sequía en Málaga' → \
['Málaga']). Un elemento por entidad, tal como aparece. NO son entidades los \
partidos o grupos (van en groups_or_parties), ni las personas (speakers / \
mentioned_persons), ni la provincia de elección de los diputados \
(constituencies). La entidad se mantiene TAMBIÉN dentro de semantic_query: es el \
tema. Null si el tema no nombra ninguna entidad concreta (un tema de nombre \
común como 'vivienda pública' no es una entidad).
- entities_mode: cómo combinar varias entidades: 'all' si deben aparecer todas \
(conjunción 'y'/'e', o una sola entidad); 'any' si basta con una (disyunción \
'o'/'u').
- speaker_title: el cargo del orador cuando se le nombra por su cargo en lugar de \
por su nombre (p. ej. 'ministra de economía'). El término genérico \
'diputados'/'diputadas' NO es un cargo. Null en otro caso.
- constituencies: TODAS las provincias por las que deben estar electos los \
diputados que intervienen, cuando la consulta filtra por circunscripción \
('diputados de Cádiz' → ['Cádiz'], 'diputados del PSOE por Málaga' → \
['Málaga']), un elemento por provincia, con el nombre propio de la provincia: \
convierte los gentilicios ('diputados malagueños' → ['Málaga']). NO es un lugar \
del que trate la intervención ('sobre la sequía en Málaga' → tema, no \
circunscripción) ni un lugar que califique otra cosa ('la carga de trabajo en \
Navantia de diputados de Cádiz' → el tema es la carga de trabajo en Navantia y \
la circunscripción es Cádiz). Null si no se filtra por circunscripción.
- groups_or_parties: TODOS los grupos parlamentarios o partidos políticos como \
filtro, un elemento por grupo/partido (p. ej. 'del PSOE y del PP' → ['PSOE', \
'PP']; 'del Grupo Socialista' → ['Grupo Socialista']). Si la consulta usa una \
categoría ideológica o de bloque en vez de nombres concretos ('la izquierda', \
'los partidos de derecha', 'los independentistas', 'los nacionalistas'), NO la \
expandas a partidos: emite el término de la categoría tal cual como un único \
elemento (p. ej. 'los partidos de izquierda' → ['izquierda']). Null si no hay.
- date_from / date_to: rango de fechas en formato ISO YYYY-MM-DD. Resuelve las \
expresiones relativas ('el último año', 'últimos tres meses', 'en 2024') tomando \
como fecha actual {today}. Null si no hay restricción temporal.
- lang, legislature: solo si se indican explícitamente.
"""


class LLMQueryParser:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._structured = None

    def _model(self):
        if self._structured is None:
            from qhld_ai.infrastructure.llm.factory import create_llm_from_env

            overrides = {}
            if self.settings.query_parser_llm_provider:
                overrides["llm_provider"] = self.settings.query_parser_llm_provider
            if self.settings.query_parser_llm_model:
                overrides["llm_model"] = self.settings.query_parser_llm_model
            llm_settings = self.settings.model_copy(update=overrides)
            chat = create_llm_from_env(llm_settings)
            self._structured = chat.with_structured_output(ParsedQuery)
        return self._structured

    def parse(self, query: str, today: date) -> ParsedQuery:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=_SYSTEM.format(today=today.isoformat())),
            HumanMessage(content=query),
        ]
        return self._model().invoke(messages)


@_register("llm")
def create(settings: Settings) -> LLMQueryParser:
    return LLMQueryParser(settings)
