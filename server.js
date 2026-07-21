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

You are scanning a live futures trading stream VOD by "Trading Patiently" (Bam) — a futures day trader who streams ES/NQ live every morning at 9:30am ET using ICT concepts on prop firm accounts. These clips go on TikTok, YouTube Shorts, Instagram Reels, and all short-form platforms.

---

## YOUR ONLY JOB
Find the 6-10 moments in this transcript that will stop a complete stranger mid-scroll in the first 3 seconds and make them think "I NEED to know what happens next."

The human brain's Reticular Activating System (RAS) filters thousands of stimuli per second. Your job is to find moments that break through that filter instantly — not by being interesting, but by triggering an unresolved tension the brain must close.

---

## THE 5 COGNITIVE TRIGGERS — score every clip against these

The best clips hit multiple triggers simultaneously. The more triggers a moment hits, the stronger the hook.

**1. NOVELTY / PATTERN DISRUPTION**
Something unexpected, jarring, or different that breaks the pattern of a normal trading stream. A sudden loud reaction, price doing something nobody expected, Bam saying something that contradicts conventional wisdom. The brain is hardwired to notice the unexpected.

**2. PERSONAL RELEVANCE**
The viewer immediately thinks "this is for me." Prop firm traders stop scrolling when they hear: drawdown, consistency rule, funded account, passing an eval, blown account, stop hunt. These words trigger immediate identification.

**3. CURIOSITY GAP**
The space between what the viewer knows and what they want to know. Bam hints at something valuable, surprising, or important without revealing it yet. "Watch what happens here." "This is why most traders fail." "I've never seen this before." The brain craves closure and stays to fill the gap.

**4. EMOTIONAL RESONANCE**
Mirror neurons fire when we see genuine emotion. Bam celebrating a win, frustrated after a stop, shocked by price action, hyped when a prediction plays out — the viewer literally feels what he feels. Raw, unscripted emotion always outperforms polished delivery.

**5. AUTHORITY BIAS**
Specific data, credentials, or results that signal expertise instantly. "I hold 20 funded accounts." "I just made $X in 2 minutes." "This setup has worked every single day this week." Numbers and specificity create credibility in seconds.

---

## CLIP STRUCTURE — Hook → Value → Payoff → CTA

**HOOK (0-3 seconds):** The pattern disruption. Must trigger the RAS immediately. An emotional outburst, a bold statement, an unresolved trade, a contrarian claim, a shocking number. Must work for someone who has never seen Bam before and knows nothing about him.

**VALUE / RETAIN (3-20 seconds):** Delivers on the hook's promise while building tension. The viewer must NEED to see what happens next. A trade in progress, a prediction not yet confirmed, a concept building toward a reveal. This is where viewers decide to stay or leave.

**PAYOFF (20+ seconds):** The satisfying moment that closes the loop. Target hit, prediction confirmed, reaction lands, reveal delivered. Must feel earned — the reward for watching.

**CTA (end):** Natural, soft, energy-matched to the clip. Never salesy. Never forced. Drives to Bam's YouTube live stream where they can watch him trade live daily.

---

## WHAT TO LOOK FOR IN THE TRANSCRIPT

Highest priority moments:
- Bam IN a live trade with the outcome unknown — pure unresolved tension
- Bam making a specific price prediction BEFORE it happens, then it plays out
- Sudden emotional reactions — celebration, frustration, shock, disbelief
- Bold contrarian statements about trading, prop firms, or markets
- Near-drawdown or near-stop-out situations with visible stakes
- A specific, punchy ICT concept explained in one sentence with a live example
- Any moment where price does exactly what Bam said — or the complete opposite at the worst time

---

## SCORING (1-10)
Score purely on scroll-stopping power. Ask: would a stranger who has never seen Bam stop scrolling in the first 3 seconds?

Count how many of the 5 cognitive triggers this moment hits:
- Hits 4-5 triggers: 9-10
- Hits 3 triggers: 7-8
- Hits 2 triggers: 5-6
- Hits 1 trigger: do not include

---

## OUTPUT
Convert timestamps to total seconds from start of video.

CRITICAL: "opening_words", "hook_suggestion", and "payoff" must be Bam's EXACT verbatim words from the transcript, copied word for word. Do not paraphrase. Do not rewrite. The editor uses these to search and find the exact moment in the video.

For each clip return:
- "title": punchy TikTok-style caption that opens a curiosity gap, max 8 words, no punctuation
- "type": one of "reaction", "prediction", "teaching", "prop_firm", "entry_exit"
- "start_seconds": integer (start just before the hook triggers)
- "end_seconds": integer (after the payoff closes the loop)
- "viral_score": integer 1-10
- "triggers": array of which cognitive triggers this clip hits e.g. ["emotional_resonance", "curiosity_gap"]
- "why_viral": one sentence — what specific open loop does this create and what makes the viewer unable to scroll past?
- "opening_words": Bam's EXACT verbatim first words at start_seconds (10-15 words) so editor can find the spot instantly
- "hook_suggestion": Bam's EXACT verbatim words that serve as the hook
- "payoff": Bam's EXACT verbatim words at the payoff moment
- "cta_suggestion": a soft, natural CTA (3-6 words max per the framework) that fits this clip's energy and drives to Bam's YouTube live stream

Also return "stream_title": a curiosity-gap title for the stream highlight reel — hints at something valuable without revealing it, makes a trader think "I need to watch this."

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
