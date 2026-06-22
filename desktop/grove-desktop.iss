; GROVEE Desktop — Inno Setup script
; Build: desktop\build-desktop-package.ps1 (sets STAGING path)

#ifndef STAGING
  #define STAGING "..\desktop-staging"
#endif

#define MyAppName "GROVEE Desktop"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "GROVEE"
#define MyAppURL "https://theicd.github.io/GROVEEMODEL/docs/"
#define MyAppExeName "Start-GroveDesktop.bat"

[Setup]
AppId={{A7B3C9D1-E4F2-4A8B-9C0D-1E2F3A4B5C6D}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
DefaultDirName={localappdata}\GROVEE\Desktop
DefaultGroupName=GROVEE
DisableProgramGroupPage=yes
OutputBaseFilename=GroveDesktop-Setup-{#MyAppVersion}
OutputDir=..\public\plugins
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\Start-GroveDesktop.bat
SetupIconFile=compiler:SetupClassicIcon.ico

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[CustomMessages]
english.WelcomeLabel2=This will install GROVEE Desktop on your computer.%n%n• Local interface in your browser%n• Local search (OpenSERP)%n%nAI models download on first use from the internet. You can change the install folder on the next page.
english.FinishedLabel=GROVEE is installed.%n%nClick the GROVEE icon on your Desktop to start.%n%nKeep the launcher window open while using the app.

[Tasks]
Name: "desktopicon"; Description: "Create a &Desktop icon"; GroupDescription: "Additional icons:"; Flags: checkedonce
Name: "launch"; Description: "Launch GROVEE when setup exits"; GroupDescription: "After install:"; Flags: checkedonce

[Files]
Source: "{#STAGING}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\GROVEE"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{autodesktop}\GROVEE"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon; WorkingDir: "{app}"

[Run]
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\Install-OpenSerp.ps1"" -SearchDir ""{app}\search"""; StatusMsg: "Downloading search engine (OpenSERP)..."; Flags: waituntilterminated
Filename: "{app}\{#MyAppExeName}"; Description: "Start GROVEE"; Flags: nowait postinstall skipifsilent; Tasks: launch

[UninstallDelete]
Type: filesandordirs; Name: "{app}"

[Code]
procedure CurStepChanged(CurStep: TSetupStep);
var
  SearchDir, UninstallPs1: String;
begin
  if CurStep = ssPostInstall then
  begin
    SearchDir := ExpandConstant('{app}\search');
    UninstallPs1 := ExpandConstant('{app}\Uninstall-GroveDesktop.ps1');
    if not FileExists(UninstallPs1) then
    begin
      SaveStringToFile(UninstallPs1,
        '# GROVEE Desktop uninstall helper' + #13#10 +
        'Remove-Item -Recurse -Force -ErrorAction SilentlyContinue ''' +
        ExpandConstant('{app}') + '''' + #13#10, False);
    end;
  end;
end;
