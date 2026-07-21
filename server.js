import express from 'express';
import multer from 'multer';
import ffmpegPath from '@ffmpeg-installer/ffmpeg';
import ffmpeg from 'fluent-ffmpeg';
import fs from 'fs';
import path from 'path';
import os from 'os';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

ffmpeg.setFfmpegPath(ffmpegPath.path);

const app = express();
const PORT = process.env.PORT || 3000;

// Serve static frontend
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json({ limit: '1mb' }));

// Multer — store uploaded video/audio in /tmp, no size limit
const upload = multer({
  dest: os.tmpdir(),
  limits: { fileSize: Infinity },
});

// ── helpers ──────────────────────────────────────────────────────────────────

function getAudioDuration(filePath) {
  return new Promise((resolve, reject) => {
    ffmpeg.ffprobe(filePath, (err, meta) => {
      if (err) return reject(err);
      resolve(meta.format.duration || 0);
    });
  });
}

function extractAudioChunk(inputPath, outputPath, startSec, durationSec) {
  return new Promise((resolve, reject) => {
    ffmpeg(inputPath)
      .setStartTime(startSec)
      .duration(durationSec)
      .noVideo()
      .audioChannels(1)
      .audioFrequency(16000)
      .audioCodec('libmp3lame')
      .audioBitrate('32k')
      .on('error', reject)
      .on('end', resolve)
      .save(outputPath);
  });
}

async function transcribeChunk(filePath, offsetSeconds, openaiKey) {
  const fileBuffer = fs.readFileSync(filePath);
  const boundary = '----WB' + Date.now();

  const pre = Buffer.from(
    `--${boundary}\r\n` +
    `Content-Disposition: form-data; name="file"; filename="chunk.mp3"\r\n` +
    `Content-Type: audio/mpeg\r\n\r\n`
  );
  const post = Buffer.from(
    `\r\n--${boundary}\r\n` +
    `Content-Disposition: form-data; name="model"\r\n\r\nwhisper-1\r\n` +
    `--${boundary}\r\n` +
    `Content-Disposition: form-data; name="response_format"\r\n\r\nverbose_json\r\n` +
    `--${boundary}\r\n` +
    `Content-Disposition: form-data; name="timestamp_granularities[]"\r\n\r\nsegment\r\n` +
    `--${boundary}--\r\n`
  );

  const res = await fetch('https://api.openai.com/v1/audio/transcriptions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${openaiKey}`,
      'Content-Type': `multipart/form-data; boundary=${boundary}`,
    },
    body: Buffer.concat([pre, fileBuffer, post]),
  });

  if (!res.ok) {
    const err = await res.text();
    let msg = `Whisper error (${res.status})`;
    try { msg = JSON.parse(err)?.error?.message || msg; } catch {}
    throw new Error(msg);
  }

  const data = await res.json();
  return (data.segments || []).map(seg => ({
    start: seg.start + offsetSeconds,
    end: seg.end + offsetSeconds,
    text: seg.text,
  }));
}

function formatTime(s) {
  s = Math.floor(s);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h > 0) return `${h}:${String(m).padStart(2,'0')}:${String(sec).padStart(2,'0')}`;
  return `${m}:${String(sec).padStart(2,'0')}`;
}

// ── routes ───────────────────────────────────────────────────────────────────

// Health check
app.get('/health', (req, res) => res.json({ status: 'ok' }));

// Upload video/audio → extract audio → transcribe → return transcript
app.post('/api/transcribe', upload.single('file'), async (req, res) => {
  const openaiKey = process.env.OPENAI_API_KEY;
  if (!openaiKey) return res.status(500).json({ error: 'OPENAI_API_KEY not configured.' });
  if (!req.file) return res.status(400).json({ error: 'No file uploaded.' });

  const inputPath = req.file.path;
  const sessionId = Date.now().toString();
  const chunkPaths = [];

  try {
    // Get duration
    const duration = await getAudioDuration(inputPath);
    if (!duration) throw new Error('Could not read file duration. Make sure the file is a valid video or audio file.');

    // Split into 15-min chunks (at 32kbps mono = ~3.5MB each, well under Whisper 25MB limit)
    const CHUNK_SECONDS = 15 * 60;
    const numChunks = Math.max(1, Math.ceil(duration / CHUNK_SECONDS));

    let allSegments = [];

    for (let i = 0; i < numChunks; i++) {
      const startSec = i * CHUNK_SECONDS;
      const chunkPath = path.join(os.tmpdir(), `chunk_${sessionId}_${i}.mp3`);
      chunkPaths.push(chunkPath);

      await extractAudioChunk(inputPath, chunkPath, startSec, CHUNK_SECONDS);
      const segments = await transcribeChunk(chunkPath, startSec, openaiKey);
      allSegments = allSegments.concat(segments);
    }

    const transcript = allSegments
      .map(s => `[${formatTime(s.start)}] ${s.text.trim()}`)
      .join('\n');

    if (transcript.length < 100) {
      throw new Error('Transcription returned too little text. Make sure the video has clear speech audio.');
    }

    return res.json({ transcript, duration: Math.round(duration) });

  } catch (err) {
    return res.status(500).json({ error: err.message || 'Transcription failed.' });
  } finally {
    // Cleanup temp files
    try { fs.unlinkSync(inputPath); } catch {}
    chunkPaths.forEach(p => { try { fs.unlinkSync(p); } catch {} });
  }
});

// Analyze transcript → return viral clips
app.post('/api/analyze', async (req, res) => {
  const anthropicKey = process.env.ANTHROPIC_API_KEY;
  if (!anthropicKey) return res.status(500).json({ error: 'ANTHROPIC_API_KEY not configured.' });

  const { transcript } = req.body;
  if (!transcript || transcript.length < 100) {
    return res.status(400).json({ error: 'Transcript too short or missing.' });
  }

  try {
    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': anthropicKey,
        'anthropic-version': '2023-06-01',
      },
      body: JSON.stringify({
        model: 'claude-sonnet-4-6',
        max_tokens: 4000,
        messages: [{
          role: 'user',
          content: `You are an elite short-form content strategist. Your entire job is to identify the highest-performing clip moments from a YouTube live trading stream and give a video editor everything they need to turn them into algorithm-optimized TikToks and YouTube Shorts.

## WHO BAM IS
"Trading Patiently" (Bam) is a futures day trader who:
- Trades ES and NQ futures live on stream every morning at 9:30am ET
- Uses ICT concepts: order blocks, fair value gaps, liquidity sweeps, displacement, market structure, killzones
- Trades prop firm funded accounts (Topstep, MyFundedFutures, Lucid, Alpha Futures, FundedSeat, etc.)
- Core message: "I create patient and profitable traders"
- Authority: holds 20 funded accounts simultaneously, pays himself from prop firm payouts

## TARGET AUDIENCE
Futures day traders grinding prop firm evaluations or managing funded accounts who know ICT/SMC concepts. They stop scrolling for live proof, raw authenticity, moments that mirror their experience, ICT-specific language, and prop firm pain points.

## ALGORITHM OBJECTIVE
Optimize for: scroll-stop rate (first 2-3 seconds), completion rate (60%+), engagement (comments, shares, saves).

## VIRAL STRUCTURE — Hook (Promised Payoff) → Retain → Reward → CTA

HOOK (0-3 sec): Proof hook, identity hook, contrarian hook, tension hook, or pattern interrupt. Must promise a specific payoff and work with zero context.

RETAIN (3 sec → payoff): Trade open with unknown outcome, setup narrated as price moves, prediction made and price approaching level, concept explained building toward conclusion.

REWARD: Target hit after live call, prediction confirmed, ICT concept clicking, authentic emotional reaction, prop firm milestone.

CTA: Drive to YouTube live stream. "I do this live every morning on YouTube — link in bio" / "Come watch me apply this live — I stream every day at 9:30" / "Watch me manage 20 funded accounts live on YouTube"

## CLIP TYPES (ranked by viral potential)
1. PREDICTION CONFIRMED — calls move before it happens, plays out on stream
2. REAL-TIME TRADE ENTRY — entry called live, outcome unknown, ends with result
3. PROP FIRM MOMENT — near-drawdown survival, passing eval, payout hit
4. RAW EMOTIONAL REACTION — unscripted win/loss reaction
5. ICT CONCEPT IN ACTION — concept explained in real time with clear aha moment

## SCORING (1-10)
Hook strength 30%, retention arc 25%, payoff quality 25%, niche specificity 20%.
Penalize: resolves immediately (cap 5), generic without ICT/prop firm specificity (cap 4).

## TRANSCRIPT
${transcript.slice(0, 18000)}

You are identifying HIGH-POTENTIAL YouTube Shorts from a trading livestream by "Trading Patiently" (Bam) — a futures day trader who streams ES/NQ live every morning using ICT concepts on prop firm accounts.

## YOUR PRIORITY
Your priority is NOT educational value. Your priority is finding moments that naturally make people want to keep watching.

## HOOK FORMULA
A good hook should:
- Introduce the topic immediately
- Leave the viewer with ONE clear unanswered question
- Never spoil the outcome
- Make the viewer NEED to know what happens next

Ask yourself: "After the first 3-5 seconds, what question is living in the viewer's mind?" If there isn't a clear unanswered question, the clip is weak.

## HOOK CHECKLIST
✓ Introduces the topic immediately
✓ Creates ONE unanswered question
✓ Doesn't reveal the outcome
✓ The rest of the clip naturally answers that question
✓ Creates curiosity without requiring lots of context

The first spoken sentence should ideally be able to serve as the opening of a Short.

## PRIORITIZE moments where Bam:
- Predicts something
- Says "Hopefully...", "We'll see...", "I don't know...", "This might...", "I don't like this..."
- Is uncertain
- Makes a difficult decision
- Reacts emotionally
- Regrets something
- Laughs
- Gets frustrated
- Talks to chat
- Says something funny or unexpected
- Makes a mistake
- Experiences tension before the outcome
- Shows personality

## AVOID moments that:
- Start with greetings
- Spend too long explaining concepts
- Need lots of previous context
- Reveal the outcome immediately
- Are mostly educational with little emotion
- Have no clear story

## CLIP STRUCTURE — every clip must follow this arc
Question → Escalation → Payoff

The opening creates curiosity. The middle increases tension. The ending answers the original question.

## SCORING
Only recommend clips that score at least 8/10 for hook strength and have a complete beginning, middle, and ending. Be extremely selective. It is better to recommend 3 excellent clips than 15 average ones.

When choosing between an educational clip and a personality-driven, emotional, humorous, tense, or story-based clip, ALWAYS choose the story-based clip.

## OUTPUT
Convert timestamps to total seconds from start of video.

CRITICAL: "opening_words" and "suggested_hook" must be Bam's EXACT verbatim words from the transcript, copied word for word. The editor uses these to find the exact moment in the video. Do not paraphrase. Do not rewrite.

For each clip return:
- "title": punchy Short-style caption, max 8 words, no punctuation
- "type": one of "reaction", "prediction", "tension", "mistake", "personality", "entry_exit"
- "start_seconds": integer
- "end_seconds": integer
- "hook_strength": integer 1-10 (only include if 8+)
- "virality_potential": integer 1-10
- "why_clip": one sentence — why this moment is worth clipping
- "unanswered_question": the exact question living in the viewer's mind after the first 3-5 seconds
- "opening_words": Bam's EXACT verbatim first words at start_seconds (10-15 words) so editor can find the spot instantly
- "suggested_hook": Bam's EXACT verbatim words that serve as the opening hook
- "story_structure": one sentence each for Question, Escalation, and Payoff — what happens in each phase

Also return "stream_title": a curiosity-gap title for this stream's highlight reel.

Return ONLY valid JSON, no markdown:
{"stream_title": "...", "clips": [...]}`,
        }],
      }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      return res.status(response.status).json({ error: err?.error?.message || 'Anthropic API error' });
    }

    const data = await response.json();
    const raw = data.content
      .filter(b => b.type === 'text')
      .map(b => b.text)
      .join('')
      .replace(/```json|```/g, '')
      .trim();

    let parsed;
    try { parsed = JSON.parse(raw); }
    catch {
      const match = raw.match(/\{[\s\S]*\}/);
      if (match) parsed = JSON.parse(match[0]);
      else return res.status(500).json({ error: 'Unexpected AI response format.' });
    }

    return res.json(parsed);

  } catch (err) {
    return res.status(500).json({ error: err.message || 'Analysis failed.' });
  }
});

app.listen(PORT, () => console.log(`Clip Finder running on port ${PORT}`));
