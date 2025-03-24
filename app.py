import os
import io
import torch
import torchaudio
import gradio as gr
import numpy as np
from huggingface_hub import hf_hub_download
import tempfile
from scipy.io import wavfile

# Import PyTorch implementation from CSM submodule
from generator import load_csm_1b, Segment

# Disable Triton compilation
os.environ["NO_TORCH_COMPILE"] = "1"

class ModelManager:
    """Simple class to manage model loading and caching"""
    def __init__(self):
        self.MODEL = None

    def get_model(self, backend):
        if self.MODEL is None:
            print(f"Loading model on {backend} backend...")
            self.MODEL = load_csm_1b(backend)
        return self.MODEL

# Default prompts are available at https://hf.co/sesame/csm-1b
prompt_filepath_conversational_a = hf_hub_download(
    repo_id="sesame/csm-1b",
    filename="prompts/conversational_a.wav"
)
prompt_filepath_conversational_b = hf_hub_download(
    repo_id="sesame/csm-1b",
    filename="prompts/conversational_b.wav"
)

SPEAKER_PROMPTS = {
    "conversational_a": {
        "text": (
            "like revising for an exam I'd have to try and like keep up the momentum because I'd "
            "start really early I'd be like okay I'm gonna start revising now and then like "
            "you're revising for ages and then I just like start losing steam"
        ),
        "audio": prompt_filepath_conversational_a
    },
    "conversational_b": {
        "text": (
            "like a super Mario level. Like it's very like high detail. And like, once you get "
            "into the park, it just like, everything looks like a computer game and they have all "
            "these, like, you know, if, if there's like a, you know, like in a Mario game"
        ),
        "audio": prompt_filepath_conversational_b
    }
}

SPACE_INTRO_TEXT = """\
# Sesame CSM 1B Gradio App

Generate conversations or monologues using CSM 1B (Conversational Speech Model).

For each speaker, you can select:
- **Random Voice**: Let the model generate a random voice (default)
- **Conversational A/B**: Use the standard packaged voices that come with the model
- **Upload Voice**: Upload your own reference audio sample

The model supports up to 10 speakers (numbered 0-9) for conversations.
"""

DEFAULT_CONVERSATION = """\
[
  {"speaker_id": 0, "text": "These voice synthesis models are absolutely mind-blowing for creating memes."},
  {"speaker_id": 1, "text": "For real! You can make any character say anything now."},
  {"speaker_id": 2, "text": "I've been using them to create the most epic voiceovers for my social media content."},
  {"speaker_id": 0, "text": "The best part is how natural they sound. Perfect for those unexpected crossover memes."},
  {"speaker_id": 1, "text": "Exactly! The dankest memes are when you combine voices with visuals nobody expected."}
]
"""

DEFAULT_MONOLOGUE = """\
[
  "I'll be reading a haiku about dank memes for you today.",
  "Internet humor",
  "Distorted ironic jokes",
  "Scrolling endlessly"
]
"""

def load_prompt_audio(audio_path, sample_rate):
    if audio_path is None:
        return torch.tensor([])

    audio_tensor, sr = torchaudio.load(audio_path)
    audio_tensor = audio_tensor.squeeze(0)
    # Resample if needed
    if sr != sample_rate:
        audio_tensor = torchaudio.functional.resample(audio_tensor, orig_freq=sr, new_freq=sample_rate)
    return audio_tensor

def prepare_prompt(text, speaker, audio_path, sample_rate):
    if audio_path is None or audio_path == "":
        return Segment(text=text, speaker=speaker)

    audio_tensor = load_prompt_audio(audio_path, sample_rate)
    if audio_tensor.numel() == 0:  # Empty tensor
        return Segment(text=text, speaker=speaker)
    else:
        return Segment(text=text, speaker=speaker, audio=audio_tensor)

def get_backend():
    """Automatically select the best available backend"""
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

# No longer needed - using torchaudio's direct MP3 saving capability
# def mp3_download function removed

def insert_pause(audio_segments, sample_rate, pause_duration_ms=150):
    """Insert pauses between audio segments"""
    if not audio_segments:
        return torch.tensor([])

    pause_samples = int(sample_rate * pause_duration_ms / 1000)
    device = audio_segments[0].device if audio_segments else torch.device('cpu')
    silence = torch.zeros(pause_samples, device=device)

    combined_segments = []
    for i, segment in enumerate(audio_segments):
        if segment.device != device:
            segment = segment.to(device)
        combined_segments.append(segment)
        if i < len(audio_segments) - 1:
            combined_segments.append(silence)

    return torch.cat(combined_segments, dim=0)

def generate_conversation(model_manager, speakers, texts, temperature=0.9, topk=50, max_audio_length_ms=10000, pause_duration_ms=150):
    """Generate a conversation with multiple speakers"""
    backend = get_backend()
    generator = model_manager.get_model(backend)
    sample_rate = generator.sample_rate

    prompt_segments = []
    for speaker_id, voice_type, text, audio in speakers:
        if voice_type != "random_voice":
            if voice_type in SPEAKER_PROMPTS:
                prompt_text = SPEAKER_PROMPTS[voice_type]["text"]
                prompt_audio = SPEAKER_PROMPTS[voice_type]["audio"]
            else:  # "upload_voice"
                prompt_text = text
                prompt_audio = audio

            if prompt_audio:
                prompt = prepare_prompt(prompt_text, speaker_id, prompt_audio, sample_rate)
                prompt_segments.append(prompt)

    generated_segments = []
    for speaker_id, text in texts:
        audio_tensor = generator.generate(
            text=text,
            speaker=speaker_id,
            context=prompt_segments + generated_segments,
            max_audio_length_ms=max_audio_length_ms,
            temperature=temperature,
            topk=topk,
        )
        generated_segments.append(Segment(text=text, speaker=speaker_id, audio=audio_tensor))

    audio_segments = [seg.audio for seg in generated_segments]
    all_audio = insert_pause(audio_segments, sample_rate, pause_duration_ms=pause_duration_ms)

    # Create temporary MP3 file
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_mp3:
        mp3_path = temp_mp3.name

    # Use torchaudio to save as MP3 directly
    audio_float = all_audio.unsqueeze(0).cpu()  # Add channel dimension and ensure on CPU
    torchaudio.save(mp3_path, audio_float, sample_rate, format="mp3")

    return mp3_path

def conversation_app():
    # Create a model manager instance
    model_manager = ModelManager()

    with gr.Blocks() as app:
        gr.Markdown(SPACE_INTRO_TEXT)

        with gr.Tab("Conversation"):
            gr.Markdown("### Configure Speakers (0-9)")

            # Create a list to hold all speaker configuration components
            speaker_configs = []
            visible_speakers = gr.State(list(range(3)))  # Start with 3 visible speakers (0, 1, 2)

            # Container for speaker configurations
            with gr.Column() as speakers_container:
                # This will be populated dynamically
                pass

            # Add/Remove speaker buttons
            with gr.Row():
                add_speaker_btn = gr.Button("Add Speaker", variant="secondary")
                remove_speaker_btn = gr.Button("Remove Speaker", variant="secondary")

            # Function to create speaker UI components
            def create_speaker_components(i, is_visible=True):
                with gr.Accordion(f"Speaker {i}", open=True, visible=is_visible) as speaker_accordion:
                    speaker_voice = gr.Dropdown(
                        choices=["random_voice", "conversational_a", "conversational_b", "upload_voice"],
                        label=f"Speaker {i} Voice",
                        value="conversational_a" if i == 0 else "conversational_b" if i == 1 else "random_voice",
                        elem_id=f"speaker{i}_voice"  # Add consistent element ID
                    )

                    # Make upload controls a separate section that appears below the dropdown
                    with gr.Column(visible=False) as upload_col:
                        speaker_text = gr.Textbox(
                            label=f"Speaker {i} Reference Text",
                            lines=2,
                            elem_id=f"speaker{i}_text"  # Add consistent element ID
                        )
                        speaker_audio = gr.Audio(
                            label=f"Speaker {i} Reference Audio",
                            type="filepath",
                            elem_id=f"speaker{i}_audio"  # Add consistent element ID
                        )

                    # Add change event to show/hide upload controls
                    speaker_voice.change(
                        lambda x: gr.update(visible=(x == "upload_voice")),
                        inputs=[speaker_voice],
                        outputs=[upload_col],
                        api_name=f"toggle_speaker{i}_upload"  # Descriptive API name
                    )

                return (speaker_voice, speaker_text, speaker_audio, upload_col, speaker_accordion)

            # Create all 10 possible speaker components (0-9)
            all_speaker_components = []
            for i in range(10):
                components = create_speaker_components(i, is_visible=(i < 3))  # Only first 3 visible by default
                all_speaker_components.append(components)
                speaker_configs.append(components[:3])  # Only store voice, text, audio for processing

            # Function to update speaker visibility
            def update_speaker_visibility(visible_speakers_list):
                updates = []
                for i in range(10):
                    is_visible = i in visible_speakers_list
                    updates.append(gr.update(visible=is_visible))
                return updates

            # Function to add a speaker
            def add_speaker(visible_list):
                if len(visible_list) < 10:
                    # Add the next speaker in sequence
                    next_speaker = max(visible_list) + 1 if visible_list else 0
                    if next_speaker < 10:
                        visible_list.append(next_speaker)
                return visible_list, *update_speaker_visibility(visible_list)

            # Function to remove a speaker
            def remove_speaker(visible_list):
                if visible_list:
                    # Remove the last speaker
                    visible_list.pop()
                return visible_list, *update_speaker_visibility(visible_list)

            # Connect button events
            accordion_outputs = [comp[4] for comp in all_speaker_components]
            add_speaker_btn.click(
                add_speaker,
                inputs=[visible_speakers],
                outputs=[visible_speakers] + accordion_outputs,
                api_name="add_conversation_speaker"  # Descriptive API name
            )

            remove_speaker_btn.click(
                remove_speaker,
                inputs=[visible_speakers],
                outputs=[visible_speakers] + accordion_outputs,
                api_name="remove_conversation_speaker"  # Descriptive API name
            )

            conversation_input = gr.TextArea(
                label="Enter conversation as JSON array",
                lines=10,
                value=DEFAULT_CONVERSATION,
                info="Format: [{'speaker_id': 0-9, 'text': 'message'}, ...] - Use any of the 10 speakers configured above",
                elem_id="conversation_input"  # Add consistent element ID
            )

            with gr.Row():
                temperature = gr.Slider(
                    minimum=0.1, maximum=2.0, value=0.9, step=0.1, label="Temperature",
                    elem_id="conversation_temperature"  # Add consistent element ID
                )
                topk = gr.Slider(
                    minimum=1, maximum=100, value=50, step=1, label="Top-k",
                    elem_id="conversation_topk"  # Add consistent element ID
                )

            with gr.Row():
                max_audio_length = gr.Slider(
                    minimum=1000, maximum=30000, value=10000, step=1000, label="Max Audio Length (ms)",
                    elem_id="conversation_max_length"  # Add consistent element ID
                )
                pause_duration = gr.Slider(
                    minimum=0, maximum=1000, value=150, step=10, label="Pause Between Utterances (ms)",
                    elem_id="conversation_pause"  # Add consistent element ID
                )

            generate_btn = gr.Button("Generate Conversation", variant="primary")
            # Use a gr.Audio to play the generated MP3
            audio_output = gr.Audio(
                label="Generated MP3",
                type="filepath",
                elem_id="conversation_output"  # Add consistent element ID
            )

            def process_conversation(*args):
                try:
                    visible_speakers_list = args[0]

                    # The last 4 arguments are the conversation text and generation parameters
                    conversation_json = args[-4]
                    temperature = args[-3]
                    topk = args[-2]
                    max_audio_length = args[-1]
                    pause_duration = args[-5]

                    # Extract only the visible/active speakers' configurations
                    speakers = []
                    for i, speaker_idx in enumerate(visible_speakers_list):
                        # Each speaker has 3 components stored in all_speaker_components
                        # which we access via the speaker_configs list
                        # The index in the args list starts after the visible_speakers_list (index 0)
                        voice_idx = 1 + (speaker_idx * 3)
                        voice_type = args[voice_idx]
                        ref_text = args[voice_idx + 1]
                        ref_audio = args[voice_idx + 2]
                        speakers.append((speaker_idx, voice_type, ref_text, ref_audio))

                    import json
                    conversation = json.loads(conversation_json)
                    if not isinstance(conversation, list):
                        raise ValueError("Invalid JSON format")

                    # Validate that all speaker_ids in conversation are configured
                    speaker_ids_in_convo = {int(item["speaker_id"]) for item in conversation}
                    configured_speaker_ids = {s[0] for s in speakers}

                    missing_speakers = speaker_ids_in_convo - configured_speaker_ids
                    if missing_speakers:
                        # Add default random voices for any missing speakers
                        for speaker_id in missing_speakers:
                            speakers.append((speaker_id, "random_voice", "", None))

                    texts = [(int(item["speaker_id"]), item["text"]) for item in conversation]

                    mp3_path = generate_conversation(model_manager, speakers, texts, temperature, topk, max_audio_length, pause_duration)
                    return mp3_path
                except Exception as e:
                    print(f"Error: {e}")
                    raise gr.Error(f"Generation failed: {e}")

            # Collect all inputs for the process_conversation function
            all_inputs = [visible_speakers]
            for voice, text, audio, _, _ in all_speaker_components:
                all_inputs.extend([voice, text, audio])
            all_inputs.extend([pause_duration, conversation_input, temperature, topk, max_audio_length])

            generate_btn.click(
                process_conversation,
                inputs=all_inputs,
                outputs=[audio_output],
                api_name="generate_conversation_audio"  # Descriptive API name
            )

        with gr.Tab("Monologue"):
            speaker_voice = gr.Dropdown(
                choices=["random_voice", "conversational_a", "conversational_b", "upload_voice"],
                label="Speaker Voice",
                value="conversational_a",
                elem_id="monologue_speaker_voice"  # Add consistent element ID
            )

            # Make upload controls a separate section that appears below the dropdown
            with gr.Column(visible=False) as upload:
                speaker_text = gr.Textbox(
                    label="Speaker Reference Text",
                    lines=2,
                    elem_id="monologue_speaker_text"  # Add consistent element ID
                )
                speaker_audio = gr.Audio(
                    label="Speaker Reference Audio",
                    type="filepath",
                    elem_id="monologue_speaker_audio"  # Add consistent element ID
                )

            speaker_voice.change(
                lambda x: gr.update(visible=(x == "upload_voice")),
                inputs=[speaker_voice],
                outputs=[upload],
                api_name="toggle_monologue_upload"  # Descriptive API name
            )

            monologue_input = gr.TextArea(
                label="Enter monologue as JSON array",
                lines=10,
                value=DEFAULT_MONOLOGUE,
                info="Format: [\"message\", \"message\", ...] - Simple array of text strings",
                elem_id="monologue_input"  # Add consistent element ID
            )

            with gr.Row():
                mono_temperature = gr.Slider(
                    minimum=0.1, maximum=2.0, value=0.9, step=0.1, label="Temperature",
                    elem_id="monologue_temperature"  # Add consistent element ID
                )
                mono_topk = gr.Slider(
                    minimum=1, maximum=100, value=50, step=1, label="Top-k",
                    elem_id="monologue_topk"  # Add consistent element ID
                )

            with gr.Row():
                mono_max_audio_length = gr.Slider(
                    minimum=1000, maximum=30000, value=10000, step=1000, label="Max Audio Length (ms)",
                    elem_id="monologue_max_length"  # Add consistent element ID
                )
                mono_pause_duration = gr.Slider(
                    minimum=0, maximum=1000, value=150, step=10, label="Pause Between Utterances (ms)",
                    elem_id="monologue_pause"  # Add consistent element ID
                )

            mono_generate_btn = gr.Button("Generate Monologue", variant="primary")
            mono_audio_output = gr.Audio(
                label="Generated MP3",
                type="filepath",
                elem_id="monologue_output"  # Add consistent element ID
            )

            def process_monologue(speaker_voice, speaker_text, speaker_audio,
                                  monologue_json, temperature, topk, max_audio_length, pause_duration):
                try:
                    import json
                    monologue = json.loads(monologue_json)
                    if not isinstance(monologue, list):
                        raise ValueError("Invalid JSON format")

                    speakers = [(0, speaker_voice, speaker_text, speaker_audio)]
                    texts = [(0, text) for text in monologue]  # Simple array of text strings

                    mp3_path = generate_conversation(model_manager, speakers, texts, temperature, topk, max_audio_length, pause_duration)
                    return mp3_path
                except Exception as e:
                    print(f"Error: {e}")
                    raise gr.Error(f"Generation failed: {e}")

            mono_generate_btn.click(
                process_monologue,
                inputs=[
                    speaker_voice, speaker_text, speaker_audio,
                    monologue_input, mono_temperature, mono_topk,
                    mono_max_audio_length, mono_pause_duration
                ],
                outputs=[mono_audio_output],
                api_name="generate_monologue_audio"  # Descriptive API name
            )

    return app

# Create the app
app = conversation_app()

if __name__ == "__main__":
    app.queue()
    app.launch(server_name="0.0.0.0", server_port=7860)