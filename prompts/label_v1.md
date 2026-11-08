You label Telegram posts from Ukrainian air-raid monitoring channels for an evaluation set.
Label EVERY post below. A post is RELEVANT FOR KYIV (1) if any of these holds, otherwise 0:
1. it mentions Kyiv city or Kyiv oblast (Київ, Київщина, столиця, Бровари, Бориспіль, Біла Церква,
   Вишгород, Обухів, Фастів, Бучa, Ірпінь, …);
2. it reports a threat moving toward Kyiv or Kyiv oblast (drones, missiles or aircraft heading there);
3. it reports a nationwide threat (MiG-31K / strategic aviation takeoff, ballistic launch, a massive
   combined attack) that can reach Kyiv.
Posts about other regions only, all-clear messages and general news are 0.

For each post also give its threat type: "drones", "cruise_missiles", "ballistic", "aviation",
"none" (no threat in the post).

Posts (one per line: [post_id] time | channel | text):
{posts}

Return JSON only: {{"labels": [{{"post_id": "...", "relevant": 0 or 1, "threat_type": "..."}}, ...]}}
with one entry per post above, in the same order.
