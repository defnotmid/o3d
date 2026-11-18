-- The soundtrack: an odawg song, chiptune over a live band after
-- Anamanaguchi (Endless Fantasy, 2013), 140 BPM in D major, 28 bars, its
-- sections named after the scenes.
--
-- Why: o3d's mascot is pixel art, and the video lifts it off the page into
-- 3D. Anamanaguchi's music does the same thing in sound: NES chips (a thin
-- pulse lead, a pulse arpeggio in sixteenths, a triangle bass, noise
-- drums) carried by a real band. Its decisions, followed: the chips alone
-- first, the band coming in under them all at once on a downbeat (here the
-- lift, where Oxa turns to voxels); bright major harmony on J-pop's royal
-- road (IV V iii vi: Gmaj7 A7 F#m7 Bm7), holding off the tonic until the
-- very last chord; a half-time verse, a full chorus, a breakdown back to
-- the chips alone, and the hook coming back doubled. Two voices play the
-- parity's hook in unison, a chip on the left and the strings on the
-- right: the same notes from two instruments, as the scene shows the same
-- pixels from two renderers.
--
-- The picture answers the notes (timeline.lua, video.lua): the coin blips
-- are its pops, the impacts its flashes, the kick its pulse.
--
-- Instruments: the chips are rows of our own (oscillators, noise, decays);
-- the band is odawg's sets, CC0 (virtuosity's jazz kit played as a rock
-- kit, Karoryfer's fingered bass, VSCO 2's string sections); the impact
-- and the riser are the onimite library's sfx rows (media/instruments).

local BPM = 140

-- The chords, voiced for the arpeggio (a 7th chord from the 4th octave).
local VOICE = {
  Gmaj7 = { "G4", "B4", "D5", "F#5" }, A7 = { "A4", "C#5", "E5", "G5" }, ["F#m7"] = { "F#4", "A4", "C#5", "E5" },
  Bm7 = { "B4", "D5", "F#5", "A5" }, Em7 = { "E4", "G4", "B4", "D5" }, Dmaj7 = { "D4", "F#4", "A4", "C#5" },
  D7 = { "D4", "F#4", "A4", "C5" },
}
local ROOT = { Gmaj7 = "G", A7 = "A", ["F#m7"] = "F#", Bm7 = "B", Em7 = "E", Dmaj7 = "D", D7 = "D" }

-- Each section's chords, two a bar.
local PROG = {
  title = { "Gmaj7", "A7", "F#m7", "Bm7" },
  lift = { "Gmaj7", "A7", "F#m7", "Bm7" },
  materials = { "Gmaj7", "A7", "F#m7", "Bm7", "Em7", "A7", "Dmaj7", "D7" },
  light = { "Bm7", "Bm7", "Gmaj7", "Gmaj7", "Em7", "Em7", "A7", "A7" },
  models = { "Gmaj7", "A7", "F#m7", "Bm7", "Gmaj7", "A7", "Bm7", "Bm7" },
  panels = { "Em7", "Em7", "F#m7", "F#m7", "Gmaj7", "Gmaj7", "A7", "A7" },
  parity = { "Gmaj7", "A7", "F#m7", "Bm7", "Gmaj7", "A7", "Em7", "A7" },
  outro = { "Gmaj7", "A7", "F#m7", "Bm7", "Gmaj7", "A7" },
  tail = { "Dmaj7", "Dmaj7" },
}
local ORDER = { "title", "lift", "materials", "light", "models", "panels", "parity", "outro", "tail" }

local function at(bar, quarter, sixteenth) return string.format("%d:%d:%d", bar, quarter, sixteenth) end

-- The arpeggio: each chord's notes in sixteenths up and back down, eight
-- to the half bar, the first of each a little stronger.
local SHAPE = { 1, 2, 3, 4, 3, 2, 1, 2 }
local function arp(chords, v)
  local ev = {}
  for c, name in ipairs(chords) do
    for i, k in ipairs(SHAPE) do
      local step = (c - 1) * 8 + (i - 1)
      ev[#ev + 1] = { time = at(step // 16, (step // 4) % 4, step % 4), note = VOICE[name][k], duration = "16n",
        velocity = v + (i == 1 and 0.08 or 0) }
    end
  end
  return ev
end

-- A bass in eighths on each chord's root, `oct` and an octave up every
-- other one (`bounce`) or held on the root.
local function bass(chords, oct, v, bounce)
  local ev = {}
  for c, name in ipairs(chords) do
    for i = 0, 3 do
      local step = (c - 1) * 4 + i
      local o = (bounce and i % 2 == 1) and oct + 1 or oct
      ev[#ev + 1] = { time = at(step // 8, (step // 2) % 4, (step % 2) * 2), note = ROOT[name] .. o, duration = "8n",
        velocity = v + (i == 0 and 0.08 or 0) }
    end
  end
  return ev
end

-- Held chords for the strings: each chord's notes an octave down for its
-- half bar (or as long as it repeats).
local function held(chords, v)
  local ev, c = {}, 1
  while c <= #chords do
    local n = 1
    while chords[c + n] == chords[c] do n = n + 1 end
    for _, note in ipairs(VOICE[chords[c]]) do
      local name, o = note:match("^(%D+)(%d)$")
      ev[#ev + 1] = { time = at((c - 1) // 2, ((c - 1) % 2) * 2, 0), note = name .. (tonumber(o) - 1), duration = n == 1 and "2n" or "1m",
        velocity = v }
    end
    c = c + n
  end
  return ev
end

-- The hook, two bars of eighths ("." a rest, "-" holds), and its answer.
local HOOK = "F#5 A5 B5 D6 - C#6 B5 A5 B5 - A5 F#5 E5 F#5 A5 -"
local ANSWER = "F#5 A5 B5 D6 - E6 F#6 E6 D6 - C#6 A5 B5 - - -"
local LOW = "D5 F#5 G5 B5 - A5 G5 F#5 G5 - F#5 D5 C#5 D5 F#5 -"

-- The chip voices, our own rows.
local CHIPS = {
  -- A pulse lead, an eighth of a cycle wide (the NES's thinnest), with a
  -- delayed vibrato.
  chip_lead = { nodes = {
    vib = { "Lfo", rate = 5.5, delay = 0.18, fade = 0.15 },
    env = { "Env", attack = 0.003, decay = 0.12, sustain = 0.7, release = 0.06 },
    osc = { "Osc", wave = "pulse", width = 0.125, frequency = { by = "vib", octaves = 0.006 }, amp = "env" },
  }, out = "osc", mono = true },
  -- A quarter-wide pulse for the arpeggio, plucked short.
  chip_arp = { nodes = {
    env = { "Env", attack = 0.002, decay = 0.07, sustain = 0.25, release = 0.03 },
    osc = { "Osc", wave = "pulse", width = 0.25, amp = "env" },
  }, out = "osc" },
  -- The triangle bass.
  chip_tri = { nodes = {
    env = { "Env", attack = 0.002, decay = 0.1, sustain = 0.8, release = 0.03 },
    osc = { "Osc", wave = "triangle", amp = "env" },
  }, out = "osc" },
  -- A coin: two pulse notes, the second a fourth up (the picture's pops).
  coin = { nodes = {
    env = { "Env", attack = 0.001, decay = 0.25, sustain = 0, release = 0.05 },
    step = { "Env", attack = 0.001, delay = 0.06, sustain = 1, release = 0.01 },
    osc = { "Osc", wave = "pulse", width = 0.5, frequency = { by = "step", octaves = 5 / 12 }, amp = "env" },
  }, out = "osc" },
  -- Noise drums: a kick of a falling square, a snare and a hat of noise.
  chip_kit = { pads = {
    kick = { instrument = { nodes = {
      amp = { "Decay", time = 0.09 },
      fall = { "Decay", time = 0.03 },
      osc = { "Osc", wave = "square", frequency = { base = 55, by = "fall", octaves = 2 }, amp = "amp" },
    }, out = "osc", note = "C2" }, gain_db = -4 },
    snare = { instrument = { nodes = {
      amp = { "Decay", time = 0.11 },
      noise = { "Noise" },
      out = { "Filter", "noise", type = "bandpass", frequency = 2200, q = 0.8, amp = "amp" },
    }, out = "out" } },
    hat = { instrument = { nodes = {
      amp = { "Decay", time = 0.03 },
      noise = { "Noise" },
      out = { "Filter", "noise", type = "highpass", frequency = 7000, amp = "amp" },
    }, out = "out" }, gain_db = -6 },
  } },
}

local sections = {}
for _, name in ipairs(ORDER) do
  local chords = PROG[name]
  sections[#sections + 1] = { name, duration = string.format("%dm", #chords // 2), chords = table.concat(chords, " ") }
end

-- The band's kit, a bar a string; `fill` ends a phrase with toms.
local ROCK = { kick = "x.....x.x.......", snare = "....x.......x...", hat = "x.x.x.x.x.x.x.x." }

return song { "score", key = "D major", bpm = BPM, seed = 3, gain_db = -15,
  instruments = { "instruments/sfx.json", CHIPS },
  sections = sections,
  returns = {
    room = { { "Reverb", decay = 1.1, pre_delay = 0.01 }, { "Filter", type = "highpass", frequency = 300 } },
    hall = { { "Reverb", decay = 2.2, pre_delay = 0.02 }, { "Filter", type = "highpass", frequency = 250 } },
  },
  parts = {
    -- The hook: the title's second bar, the chorus, the parity (doubled)
    -- and the outro; a lower answer through the verse.
    lead = { role = "melody", instrument = "chip_lead", gain_db = -2, pan = -0.1, sends = { hall = -16 },
      play = {
        { "title@1m", notes = "F#5 A5 B5 D6 - C#6 A5 -", subdivision = "8n", velocity = 0.6 },
        { "lift", notes = HOOK, subdivision = "8n", velocity = 0.7 },
        { "materials@2m", notes = LOW, subdivision = "8n", velocity = 0.6 },
        { "models", notes = HOOK .. " " .. ANSWER, subdivision = "8n", velocity = 0.74 },
        { "parity", notes = HOOK .. " " .. ANSWER, subdivision = "8n", velocity = 0.72 },
        { "outro", notes = HOOK, subdivision = "8n", velocity = 0.74 },
      } },
    -- The parity's other voice: the hook on the strings, right.
    twin = { role = "counter", instrument = { "Sampler", set = "vsco2/strings", attack = 0.02, release = 0.3 }, gain_db = 5, pan = 0.45,
      sends = { hall = -14 },
      play = { { "parity", notes = HOOK .. " " .. ANSWER, subdivision = "8n", velocity = 0.7 } } },
    arp = { role = "comp", instrument = "chip_arp", gain_db = 1.5, pan = 0.25, sends = { hall = -20 },
      play = (function()
        local p = {}
        for _, name in ipairs(ORDER) do
          if name ~= "tail" and name ~= "light" then
            p[#p + 1] = { name, events = arp(PROG[name], name == "panels" and 0.45 or 0.38) }
          end
        end
        p[#p + 1] = { "light@2m", events = arp({ "Em7", "Em7", "A7", "A7" }, 0.32) }
        return p
      end)() },
    -- The triangle: the chips' own bass, before the band and in the breakdown.
    tri = { role = "bass", instrument = "chip_tri", gain_db = -4,
      play = {
        { "title", events = bass(PROG.title, 3, 0.55, true) },
        { "panels", events = bass(PROG.panels, 3, 0.55, true) },
      } },
    chipdrums = { role = "drums", instrument = "chip_kit", gain_db = 1.5,
      play = {
        { "title", velocity = 0.6, steps = { hat = { "x.x.x.x.x.x.x.x.", "x.x.x.x.xxxxxxxx" }, kick = { "x.......x.......", "x.......x...x.x." } } },
        { "panels", velocity = 0.62, steps = { hat = "x.x.x.x.x.x.x.x.", kick = "x.......x.......",
          snare = { "................", "....x.......x...", "....x.......x...", "....x.......x.xx" } } },
      } },
    -- The band.
    drums = { role = "drums", instrument = { "Sampler", set = "virtuosity/drums" }, gain_db = 2, sends = { room = -12 },
      play = {
        { "lift", velocity = 0.82, humanize = 0.004, steps = { crash = { "x...............", "................" },
          kick = ROCK.kick, snare = ROCK.snare, hat = ROCK.hat } },
        { "materials", velocity = 0.74, humanize = 0.004, steps = { kick = ROCK.kick, snare = ROCK.snare, hat = ROCK.hat } },
        { "light", velocity = 0.7, humanize = 0.004, steps = { kick = "x...........x...", snare = "........x.......",
          ride = "x...x...x...x..." } },
        { "models", velocity = 0.84, humanize = 0.004, steps = { crash = { "x...............", "................" },
          kick = "x.....x.x.x.....", snare = "....x.......x...", hat = "xxxxxxxxxxxxxxxx" } },
        { "parity", velocity = 0.8, humanize = 0.004, steps = { crash = { "x...............", "................" },
          kick = ROCK.kick, snare = { ROCK.snare, ROCK.snare, ROCK.snare, "....x.......xxxx" }, hat = ROCK.hat } },
        { "outro", velocity = 0.84, humanize = 0.004, steps = { crash = { "x...............", "................", "x..............." },
          kick = ROCK.kick, snare = ROCK.snare, hat = ROCK.hat } },
        { "tail", velocity = 0.8, steps = { crash = "x...............", kick = "x..............." } },
      } },
    bass = { role = "bass", instrument = { "Sampler", set = "karoryfer/growlybass", release = 0.15 }, gain_db = -5,
      play = (function()
        local p = {}
        for _, name in ipairs({ "lift", "materials", "models", "parity", "outro" }) do
          p[#p + 1] = { name, events = bass(PROG[name], 2, 0.7, false) }
        end
        p[#p + 1] = { "light", events = held(PROG.light, 0.7) }
        p[#p + 1] = { "tail", events = { { time = 0, note = "D2", duration = "1m", velocity = 0.8 } } }
        return p
      end)() },
    strings = { role = "pad", instrument = { "Sampler", set = "vsco2/strings", attack = 0.5, release = 1.4 }, gain_db = 5,
      sends = { hall = -10 },
      play = (function()
        local p = {}
        for _, name in ipairs({ "lift", "materials", "light", "models", "panels", "outro", "tail" }) do
          p[#p + 1] = { name, events = held(PROG[name], 0.55) }
        end
        return p
      end)() },
    -- The pops: one a material, the Models' three, the panels' and the
    -- parity's numbers.
    blips = { role = "fx", instrument = "coin", gain_db = 4, pan = 0.15, sends = { hall = -14 },
      play = {
        { "materials", degrees = ". . . . 1 . 3 . 5 . 6 . 8 . . . . . . . . . . . . . . . . . . .", subdivision = "8n", velocity = 0.7, octave = 5 },
        { "models", degrees = "1 . . . 3 . . . 5 . . . . . . .", subdivision = "4n", velocity = 0.7, octave = 5 },
        { "panels", degrees = "1 . . . 3 . . . 5 . . . . . . .", subdivision = "4n", velocity = 0.66, octave = 5 },
        { "parity@2m", degrees = "1 . . . 5 . . .", subdivision = "4n", velocity = 0.72, octave = 6 },
      } },
    -- The riser into the lift and the impacts the picture flashes on.
    riser = { role = "fx", instrument = "riser", gain_db = 0,
      play = { { "title@1m", events = { { time = 0, note = "D4", duration = "1m", velocity = 0.6 } } } } },
    fx = { role = "fx", instrument = "impact", gain_db = 6,
      play = {
        { "lift", events = { { time = 0, note = "D2", duration = "2n", velocity = 0.9 } } },
        { "models", events = { { time = 0, note = "G2", duration = "2n", velocity = 0.8 } } },
        { "outro", events = { { time = 0, note = "D2", duration = "2n", velocity = 0.9 } } },
      } },
  },
  -- The lead moves left for the parity, where the strings double it on
  -- the right.
  automation = {
    ["parts/lead/pan"] = { { "title", -0.1 }, { "parity", -0.45, "step" }, { "outro", -0.1, "step" } },
    -- Levels by measurement (onimite audio parts): the breakdown and the
    -- tail are quieter mixes, the half-time verse's kit sparser.
    ["parts/arp/gain_db"] = { { "title", -1.5 }, { "lift", 0, "step" }, { "panels", -6, "step" }, { "parity", 0, "step" } },
    ["parts/strings/gain_db"] = { { "lift", 0 }, { "light", -5, "step" }, { "models", 0, "step" }, { "panels", -8, "step" },
      { "parity", 0, "step" }, { "tail", -10, "step" } },
    ["parts/bass/gain_db"] = { { "lift", 0 }, { "light", -4, "step" }, { "models", 0, "step" } },
    ["parts/drums/gain_db"] = { { "lift", 0 }, { "light", 4, "step" }, { "models", 0, "step" } },
  },
}
