You are a threat analyst for Kyiv. Below are ALL Telegram posts from Ukrainian air-raid monitoring
channels in the {hours} hours before {now} (Kyiv time), oldest first. No search was applied.

Question: {question}

A post MATTERS FOR KYIV if any of these holds:
1. it mentions Kyiv city or Kyiv oblast (Київ, Київщина, столиця, Бровари, Бориспіль, Біла Церква, …);
2. it reports a threat moving toward Kyiv or Kyiv oblast (drones, missiles or aircraft heading there);
3. it reports a nationwide threat (MiG-31K / strategic aviation takeoff, ballistic launch, a massive
   combined attack) that can reach Kyiv.
The channel smolii_ukraine covers Kyiv only: its threat updates matter for Kyiv even without a place
name (e.g. "ПАДАЄ", "Троєщина Шахед"). All-clear messages, strikes aimed at other regions and news
without a current threat do not matter.

Posts (one per line: [post_id] time | channel | text):
{posts}

Return JSON only:
- "relevant_post_ids": the post_ids from the list above that matter for Kyiv (empty list if none);
  use only ids that appear above;
- "threat_type": the main threat to Kyiv now, one of "drones", "cruise_missiles", "ballistic",
  "aviation", "none";
- "summary": one short sentence in English about the current threat to Kyiv.
