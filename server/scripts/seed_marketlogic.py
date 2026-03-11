from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


async def main() -> None:
    from app.db.seed_marketlogic import seed_marketlogic
    from app.db.session import get_sessionmaker

    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        counts = await seed_marketlogic(session)

    print("Seeded MarketLogic data:")
    for key, value in counts.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
