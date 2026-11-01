#include "SmartHelmet/SmartHelmetGameMode.h"
#include "SmartHelmet/SmartHelmetCharacter.h"

ASmartHelmetGameMode::ASmartHelmetGameMode()
{
	DefaultPawnClass = ASmartHelmetCharacter::StaticClass();
}
