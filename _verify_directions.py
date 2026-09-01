"""验证：澄清方向不足 3 个时会被补足到至少 3 个。"""
import asyncio

from src.graph import query_triage as qt


# 1. _parse_json 健壮性
assert qt._parse_json('{"ambiguous": true, "directions": "单一方向"}')["directions"] == ["单一方向"]
assert qt._parse_json('{"ambiguous": false, "directions": []}')["directions"] == []
assert qt._parse_json('```json\n{"ambiguous": true, "directions": ["a", "b", "c"]}\n```')["directions"] == ["a", "b", "c"]
print("1. _parse_json 健壮性 OK")


async def fake_expand(raw, existing, target=3):
    return list(existing)  # 模拟 LLM 补充失败


qt._expand_directions = fake_expand


async def main():
    d0 = await qt._ensure_directions("人机协同", [])
    assert len(d0) == 3, d0
    print("2. 空方向 -> 3 个（模板兜底）:", d0)

    d1 = await qt._ensure_directions("人机协同", ["人机协同的交互框架"])
    assert len(d1) == 3, d1
    print("3. 1 个方向 -> 补足到 3 个:", d1)

    d3 = await qt._ensure_directions("人机协同", ["a", "b", "c"])
    assert len(d3) == 3, d3
    print("4. 3 个方向 -> 保持 3 个:", d3)


asyncio.run(main())
print("DIRECTIONS TEST: PASSED")
