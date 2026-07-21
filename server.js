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

Identify TOP 6-10 clips. Convert timestamps to total seconds from start of video.

For each clip return:
- "title": curiosity-gap title, max 8 words, no punctuation
- "type": one of "prediction", "entry_exit", "prop_firm", "reaction", "teaching"
- "start_seconds": integer (start a few seconds before the hook)
- "end_seconds": integer
- "viral_score": integer 1-10
- "why_viral": one sentence naming the Hook → Retain → Reward arc
- "hook_suggestion": exact words or text overlay for the first 3 seconds
- "payoff": one sentence describing the climax moment the editor builds toward
- "cta_suggestion": natural CTA driving to Bam's YouTube live stream

Also return "stream_title": curiosity-gap headline for the stream highlight reel.

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
