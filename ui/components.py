from nicegui import ui
import datetime

def transcript_row(text: str, timestamp: float, is_final: bool = True):
    """Creates a row for the transcript."""
    time_str = datetime.datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
    
    with ui.row().classes('w-full items-start gap-2 mb-2'):
        ui.label(time_str).classes('text-gray-400 text-xs mt-1 min-w-[60px]')
        
        card_classes = 'p-3 rounded-lg flex-grow'
        if is_final:
            card_classes += ' bg-blue-50'
        else:
            card_classes += ' bg-gray-50 italic text-gray-500'
            
        with ui.card().classes(card_classes).style('box-shadow: none; border: 1px solid #e5e7eb'):
             ui.markdown(text).classes('text-sm')

def suggestion_card(content: str):
    """Creates a card for AI suggestions."""
    # Simple markdown parsing for the list format we asked the LLM for
    with ui.card().classes('w-full mb-4 bg-purple-50 border-l-4 border-purple-500').style('box-shadow: sm'):
        with ui.column().classes('gap-1'):
            ui.label('AI Copilot Suggestion').classes('text-xs font-bold text-purple-700 uppercase tracking-wider')
            ui.markdown(content).classes('text-sm text-gray-800')
