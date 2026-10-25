from app.parser import parse_whatsapp_export, clean_line, is_media_omitted, is_system_notice


def test_clean_line_strips_unicode_marks():
    raw = "\u200e[12/05/2023, 08:30:00\u202fAM] User: Test\u200f"
    cleaned = clean_line(raw)
    assert "\u200e" not in cleaned
    assert "\u200f" not in cleaned
    assert "\u202f" not in cleaned
    assert cleaned == "[12/05/2023, 08:30:00 AM] User: Test"


def test_android_parsing_and_counts(sample_android_chat):
    result = parse_whatsapp_export(sample_android_chat)
    assert result.user_message_count == 5
    assert result.system_message_count == 1
    assert result.media_count == 1
    assert "Me" in result.senders
    assert "Alex" in result.senders
    assert result.senders["Me"] == 3
    assert result.senders["Alex"] == 3


def test_multiline_message_handling(sample_android_chat):
    result = parse_whatsapp_export(sample_android_chat)
    # The second message from 'Me' should contain newline content
    me_msgs = [m for m in result.messages if m.sender == "Me"]
    assert len(me_msgs) == 3
    first_me = me_msgs[0]
    assert "yeah just woke up lol\nwhats the plan for today?" == first_me.content


def test_ios_bracketed_format(sample_ios_chat):
    result = parse_whatsapp_export(sample_ios_chat)
    assert result.system_message_count == 1
    assert result.user_message_count == 4
    assert result.senders["Sarah"] == 2
    assert result.senders["Me"] == 2

    # Check multiline in iOS
    me_msg = [m for m in result.messages if m.sender == "Me"][0]
    assert "hey Sarah! yes absolutely 🎉\ncan i bring a friend?" == me_msg.content


def test_media_omitted_detection():
    assert is_media_omitted("<Media omitted>")
    assert is_media_omitted("<image omitted>")
    assert is_media_omitted("video omitted")
    assert is_media_omitted("sticker omitted")
    assert is_media_omitted("This message was deleted")
    assert not is_media_omitted("Hey check out this video")


def test_system_notice_detection():
    assert is_system_notice("Messages and calls are end-to-end encrypted.")
    assert is_system_notice("John added Sarah to the group")
    assert is_system_notice("You changed the subject to Vacation")
    assert is_system_notice("Alex left")
    assert not is_system_notice("I just added some salt to the soup")
